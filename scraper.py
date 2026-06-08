#!/usr/bin/env python3
"""
Recipe Scraper Agent
--------------------
Paste a URL, and this agent fetches the page, extracts the recipe text,
then uses an LLM to:
  1. Parse ingredients into structured { amount, unit, name }
  2. Match each ingredient against ingredients_nutrition.json (fuzzy / semantic)
  3. Auto-create new nutrition entries for any ingredient not in the DB
  4. Save the complete recipe (with nutritionId per ingredient) to recipes.json

Supported LLM backends (auto-detected from env vars):
  1. GitHub Models (FREE with GitHub Copilot license)
       export GITHUB_TOKEN=<your personal access token>
  2. OpenAI
       export OPENAI_API_KEY=sk-...
  3. Anthropic
       export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python scraper.py https://example.com/recipe
  python scraper.py          # prompts for URL interactively
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict, Any

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths & model config
# ---------------------------------------------------------------------------

RECIPES_JSON   = Path(__file__).parent / "recipes.json"
NUTRITION_JSON = Path(__file__).parent / "ingredients_nutrition.json"

GITHUB_MODELS_BASE  = "https://models.inference.ai.azure.com"
GITHUB_MODELS_MODEL = "Meta-Llama-3.1-405B-Instruct"
OPENAI_MODEL        = "gpt-4o"
ANTHROPIC_MODEL     = "claude-3-5-sonnet-20241022"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}

# ---------------------------------------------------------------------------
# HTTP scraping
# ---------------------------------------------------------------------------

def fetch_html(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def extract_recipe_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "iframe", "noscript", "form"]):
        tag.decompose()

    CANDIDATES = [
        '[itemtype*="Recipe"]', '[class*="recipe-body"]', '[class*="recipeDetail"]',
        '[class*="recipe_detail"]', '[class*="recipe-content"]', '[class*="ingredients"]',
        '[id*="recipe"]', 'article', 'main', '.post-content', '.entry-content', '.article-body',
    ]
    for sel in CANDIDATES:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text
    return soup.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# LLM abstraction — one shared caller for all steps
# ---------------------------------------------------------------------------

def _detect_backend():
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", {"api_key": os.environ["OPENAI_API_KEY"], "model": OPENAI_MODEL}
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", {"api_key": os.environ["ANTHROPIC_API_KEY"]}
    if os.environ.get("GITHUB_TOKEN"):
        return "github", {"api_key": os.environ["GITHUB_TOKEN"],
                          "base_url": GITHUB_MODELS_BASE,
                          "model": GITHUB_MODELS_MODEL}
    return "none", {}


def make_llm_caller():
    """
    Returns (caller_fn, backend_label).
    caller_fn(messages: list[dict]) -> str  (always returns JSON text)
    """
    backend, kwargs = _detect_backend()

    if backend == "none":
        print(
            "\nNo API key found. Set one of:\n"
            "  export GITHUB_TOKEN=<your GitHub PAT>       # free with Copilot\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
        )
        sys.exit(1)

    labels = {
        "github":    "GitHub Models ({})".format(GITHUB_MODELS_MODEL),
        "openai":    "OpenAI ({})".format(OPENAI_MODEL),
        "anthropic": "Anthropic ({})".format(ANTHROPIC_MODEL),
    }

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=kwargs["api_key"])

        def call(messages):
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=4096,
                system=system, messages=user_msgs,
            )
            return _strip_fences(resp.content[0].text)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=kwargs["api_key"], base_url=kwargs.get("base_url"))
        model  = kwargs.get("model", OPENAI_MODEL)
        # gpt-4o supports json_object mode; Llama/others do not — use prompt only
        is_openai_model = model.startswith(("gpt-", "o1", "o3"))

        def call(messages):
            kwargs_create = dict(model=model, messages=messages,
                                 temperature=0.1, max_tokens=4096)
            if is_openai_model:
                kwargs_create["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs_create)
            return resp.choices[0].message.content

    return call, labels[backend]


def _strip_fences(raw):
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return raw


# ---------------------------------------------------------------------------
# Step 1 — Parse recipe text
# ---------------------------------------------------------------------------

_RECIPE_SYSTEM = (
    "You are a precise recipe parser that understands both Japanese and English. "
    "Always respond with valid JSON only — no markdown, no extra text."
)

_RECIPE_USER = """\
Extract the recipe from the text below (may be Japanese, English, or mixed).

Return ONLY this JSON structure:
{{
  "title": "recipe name (keep original language)",
  "servings": 4,
  "prepTime": "10 min",
  "cookTime": "30 min",
  "ingredients": [
    {{"amount": "200", "unit": "g",    "name": "鶏もも肉"}},
    {{"amount": "2",   "unit": "tbsp", "name": "soy sauce"}},
    {{"amount": "",    "unit": "",     "name": "black pepper to taste"}}
  ],
  "instructions": ["Step 1", "Step 2"]
}}

Rules:
- amount: numeric string only ("200", "1/2") — never include the unit
- unit: g, kg, ml, L, tsp, tbsp, cup, piece, sheet, pinch, clove …
  Japanese: 大さじ→tbsp, 小さじ→tsp, カップ→cup, 合→go, 本→piece, 枚→sheet
- name: keep in original language
- servings: integer or float
- prepTime / cookTime: "15 min", "1 hr", etc.

Recipe text:
{text}
"""


def parse_recipe_text(text, call):
    messages = [
        {"role": "system", "content": _RECIPE_SYSTEM},
        {"role": "user",   "content": _RECIPE_USER.format(text=text[:6000])},
    ]
    return json.loads(_strip_fences(call(messages)))


# ---------------------------------------------------------------------------
# Step 2 — Match ingredients against nutrition DB
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = (
    "You are a nutrition database matcher. "
    "Respond with valid JSON only — no markdown, no extra text."
)

_MATCH_USER = """\
Match each recipe ingredient to the best entry in the catalog below.
Use semantic judgment — ingredients may be in Japanese or English.

Examples of valid matches:
  "鶏もも肉" → Chicken Thigh
  "しょうゆ" or "醤油" → Soy Sauce
  "薄力粉"   → All-Purpose Flour
  "人参"     → Carrot
  "豚ひき肉" → Ground Pork
  "生クリーム" → Heavy Cream
  "だし"     → Dashi Stock (liquid)

Catalog  (id | English name | Japanese name):
{catalog}

Ingredients to match (one per line):
{ingredients}

Return ONLY:
{{
  "matches": [
    {{"name": "ingredient as given", "catalogId": 2,    "matchedName": "Chicken Thigh"}},
    {{"name": "mystery ingredient",  "catalogId": null, "matchedName": null}}
  ]
}}

Rules:
- catalogId must be an integer from the catalog, or null if truly absent
- Only use null when the ingredient is genuinely not in the catalog
- Partial matches, synonyms, and common-sense equivalents are all acceptable
"""


def load_nutrition_db():
    if NUTRITION_JSON.exists():
        with open(NUTRITION_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_nutrition_db(db):
    with open(NUTRITION_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _build_catalog_text(db):
    return "\n".join(
        "{} | {} | {}".format(e["id"], e["name"], e.get("nameJa", ""))
        for e in db
    )


def match_ingredients(parsed, db, call):
    """
    Adds 'nutritionId' and 'nutritionMatch' to each ingredient dict.
    Returns (updated_parsed, list_of_unmatched_names).
    """
    if not db or not parsed:
        return parsed, [i.get("name", "") for i in parsed]

    catalog_text = _build_catalog_text(db)
    names_text   = "\n".join("- {}".format(i.get("name", "")) for i in parsed if i.get("name"))

    messages = [
        {"role": "system", "content": _MATCH_SYSTEM},
        {"role": "user",   "content": _MATCH_USER.format(
            catalog=catalog_text, ingredients=names_text
        )},
    ]
    result    = json.loads(_strip_fences(call(messages)))
    match_map = {m["name"]: m for m in result.get("matches", [])}

    unmatched = []
    for ing in parsed:
        name  = ing.get("name", "")
        match = match_map.get(name, {})
        cid   = match.get("catalogId")
        if cid:
            ing["nutritionId"]    = int(cid)
            ing["nutritionMatch"] = match.get("matchedName", "")
        else:
            ing["nutritionId"]    = None
            ing["nutritionMatch"] = None
            if name:
                unmatched.append(name)

    return parsed, unmatched


# ---------------------------------------------------------------------------
# Step 3 — Generate nutrition data for unmatched ingredients
# ---------------------------------------------------------------------------

_GEN_SYSTEM = (
    "You are a nutrition expert. "
    "Respond with valid JSON only — no markdown, no extra text."
)

_GEN_USER = """\
Generate accurate nutritional data per 100g for each ingredient listed.
Use standard references (USDA, Japanese food composition tables).

Ingredients (JSON array of names):
{ingredients}

Return ONLY a JSON array — one object per ingredient:
[
  {{
    "original_name": "ingredient exactly as given in the input array",
    "name": "canonical English name",
    "nameJa": "日本語名 (empty string if unknown)",
    "category": "protein|vegetable|fruit|grain|dairy|oil|legume|nut|seasoning|baking|pantry|beverage",
    "per100g": {{
      "calories": 0,
      "protein": 0.0,
      "carbs": 0.0,
      "sugar": 0.0,
      "saturatedFat": 0.0,
      "unsaturatedFat": 0.0,
      "sodium": 0
    }}
  }}
]
"""


def create_missing_ingredients(unmatched_names, db, call):
    """
    Calls the LLM to generate nutrition for each unmatched ingredient,
    appends them to the DB, saves the file.
    Returns (original_name → new_id  map,  updated_db).
    """
    if not unmatched_names:
        return {}, db

    messages = [
        {"role": "system", "content": _GEN_SYSTEM},
        {"role": "user",   "content": _GEN_USER.format(
            ingredients=json.dumps(unmatched_names, ensure_ascii=False)
        )},
    ]
    raw         = call(messages)
    new_entries = json.loads(_strip_fences(raw))

    # Unwrap if LLM accidentally wrapped the array in an object
    if isinstance(new_entries, dict):
        new_entries = next(iter(new_entries.values()), [])

    next_id    = max((e["id"] for e in db), default=0) + 1
    name_to_id = {}

    for entry in new_entries:
        if not isinstance(entry, dict):
            continue
        orig_name = entry.pop("original_name", entry.get("name", ""))
        if not orig_name:
            continue
        entry["id"] = next_id
        next_id    += 1
        db.append(entry)
        name_to_id[orig_name] = entry["id"]

    save_nutrition_db(db)
    return name_to_id, db


# ---------------------------------------------------------------------------
# Persist recipe
# ---------------------------------------------------------------------------

def _ingredients_to_strings(parsed):
    result = []
    for ing in parsed:
        amount = ing.get("amount", "").strip()
        unit   = ing.get("unit",   "").strip()
        name   = ing.get("name",   "").strip()
        if not name:
            continue
        result.append(" ".join(p for p in [amount, unit, name] if p))
    return result


def save_to_recipes_json(recipe_data, source_url):
    if RECIPES_JSON.exists():
        with open(RECIPES_JSON, "r", encoding="utf-8") as f:
            recipes = json.load(f)
    else:
        recipes = []

    parsed = recipe_data.get("ingredients", [])

    new_recipe = {
        "id":                int(time.time() * 1000),
        "title":             recipe_data.get("title", "Untitled Recipe"),
        "servings":          recipe_data.get("servings", 2),
        "prepTime":          recipe_data.get("prepTime", ""),
        "cookTime":          recipe_data.get("cookTime", ""),
        "season":            None,
        "calories":          None,
        "protein":           None,
        "carbs":             None,
        "ingredients":       _ingredients_to_strings(parsed),
        "ingredientsParsed": parsed,   # includes nutritionId + nutritionMatch
        "instructions":      recipe_data.get("instructions", []),
        "originalUrl":       source_url,
        "lastModified":      time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }

    recipes.insert(0, new_recipe)
    with open(RECIPES_JSON, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)

    return new_recipe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else input("Paste recipe URL: ").strip()
    if not url.startswith("http"):
        print("Error: URL must start with http:// or https://")
        sys.exit(1)

    call, backend_label = make_llm_caller()
    print("\n  backend : {}".format(backend_label))

    print("\n[1/5] Fetching  {}".format(url))
    html = fetch_html(url)

    print("[2/5] Extracting recipe text...")
    text = extract_recipe_text(html)
    print("       {:,} characters extracted".format(len(text)))

    print("[3/5] Parsing recipe with LLM...")
    recipe_data        = parse_recipe_text(text, call)
    parsed_ingredients = recipe_data.get("ingredients", [])

    print("[4/5] Matching ingredients to nutrition database...")
    db = load_nutrition_db()
    parsed_ingredients, unmatched = match_ingredients(parsed_ingredients, db, call)

    for ing in parsed_ingredients:
        name  = ing.get("name", "")
        nid   = ing.get("nutritionId")
        nmatch = ing.get("nutritionMatch", "")
        if nid:
            print("       matched  : {}  ->  {} (#{})" .format(name, nmatch, nid))
        else:
            print("       new entry: {}".format(name))

    if unmatched:
        print("       Generating nutrition for {} new ingredient(s)...".format(len(unmatched)))
        name_to_id, db = create_missing_ingredients(unmatched, db, call)

        for ing in parsed_ingredients:
            if ing.get("nutritionId") is None:
                orig   = ing.get("name", "")
                new_id = name_to_id.get(orig)
                if new_id:
                    ing["nutritionId"]    = new_id
                    entry = next((e for e in db if e["id"] == new_id), {})
                    ing["nutritionMatch"] = entry.get("name", orig)
                    print("       created  : {}  ->  #{}" .format(orig, new_id))

    recipe_data["ingredients"] = parsed_ingredients

    print("[5/5] Saving to recipes.json...")
    saved = save_to_recipes_json(recipe_data, url)

    # ── Summary ────────────────────────────────────────────────────────────
    total   = len(parsed_ingredients)
    matched = sum(1 for i in parsed_ingredients if i.get("nutritionId"))
    print("\n" + "─" * 54)
    print("  Title     : {}".format(saved["title"]))
    print("  Servings  : {}".format(saved["servings"]))
    print("  Prep/Cook : {} / {}".format(saved["prepTime"], saved["cookTime"]))
    print("  Ingredients ({}):".format(total))
    for ing in parsed_ingredients:
        amount = ing.get("amount", "")
        unit   = ing.get("unit",   "")
        name   = ing.get("name",   "")
        nid    = ing.get("nutritionId")
        tag    = "  [#{}]".format(nid) if nid else "  [new]"
        print("    {:>6} {:<6}  {}{}".format(amount, unit, name, tag))
    print("─" * 54)
    print("  Nutrition matched: {}/{} ingredients".format(matched, total))
    print("  Saved to recipes.json  (id: {})\n".format(saved["id"]))


if __name__ == "__main__":
    main()
