#!/usr/bin/env python3
"""
migrate_recipes.py
------------------
One-time (re-runnable) migration that:
  1. Parses the free-text ingredient strings in every recipe into
     structured { amount, unit, name } objects
  2. Matches each ingredient against ingredients_nutrition.json
     using LLM semantic judgement (handles Japanese ↔ English)
  3. Auto-creates new nutrition entries for any unmatched ingredient
  4. Removes the manually-entered calories / protein / carbs fields
     (viewer.html now calculates these dynamically from ingredientsParsed)
  5. Saves the updated recipes.json

Safe to re-run: skips recipes whose ingredients are already fully matched.

Usage:
  python migrate_recipes.py
"""

import json
import sys
from pathlib import Path

# Reuse all infrastructure from scraper.py
from scraper import (
    make_llm_caller,
    _strip_fences,
    load_nutrition_db,
    save_nutrition_db,
    match_ingredients,
    create_missing_ingredients,
    RECIPES_JSON,
)

# ---------------------------------------------------------------------------
# Step 1 — Parse free-text ingredient strings into {amount, unit, name}
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = (
    "You are a precise ingredient parser. "
    "Respond with valid JSON only — no markdown, no extra text."
)

_PARSE_USER = """\
Parse each ingredient string into amount, unit, and name.

Input strings (JSON array):
{strings}

Return ONLY a JSON array — one object per input string, same order:
[
  {{"original": "250 g 豚ばら肉",         "amount": "250", "unit": "g",     "name": "豚ばら肉"}},
  {{"original": "1/2 cup red wine",       "amount": "1/2", "unit": "cup",   "name": "red wine"}},
  {{"original": "5 egg yolks",            "amount": "5",   "unit": "piece", "name": "egg yolks"}},
  {{"original": "Salt and pepper",        "amount": "",    "unit": "",      "name": "Salt and pepper"}},
  {{"original": "⅛ teaspoon salt",       "amount": "1/8", "unit": "tsp",   "name": "salt"}}
]

Rules:
- amount: numeric string only ("200", "1/2", "2 1/4") — no units, no words
  Convert Unicode fractions: ½→1/2, ⅓→1/3, ¼→1/4, ⅛→1/8, ¾→3/4, etc.
- unit: g, kg, ml, L, tsp, tbsp, cup, oz, lb, piece, clove, sheet, pinch, slice
  Convert words: teaspoon→tsp, tablespoon→tbsp, cup→cup, pound→lb, ounce→oz
- name: ingredient name only — no amounts, no units, no descriptors like "divided"
- Keep names in original language (Japanese is fine)
- If no amount/unit can be determined, use empty string ""
"""


def parse_ingredient_strings(strings, call):
    """
    Parse a list of ingredient strings into structured dicts.
    Returns a dict mapping original string → {amount, unit, name}.
    """
    # Process in chunks to avoid token/response-length limits
    CHUNK = 20
    chunks = [strings[i:i + CHUNK] for i in range(0, len(strings), CHUNK)]
    result = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": _PARSE_SYSTEM},
            {"role": "user",   "content": _PARSE_USER.format(
                strings=json.dumps(chunk, ensure_ascii=False)
            )},
        ]
        raw   = call(messages)
        chunk_result = json.loads(_strip_fences(raw))
        if isinstance(chunk_result, dict):
            chunk_result = next(iter(chunk_result.values()), [])
        result.extend(chunk_result)

    parsed_map = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        orig = item.get("original", "")
        if not orig:
            continue
        parsed_map[orig] = {
            "amount": item.get("amount", ""),
            "unit":   item.get("unit",   ""),
            "name":   item.get("name",   ""),
        }
    return parsed_map


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------

def needs_parse(recipe):
    return not recipe.get("ingredientsParsed")


def needs_match(recipe):
    parsed = recipe.get("ingredientsParsed", [])
    return bool(parsed) and not any(ing.get("nutritionId") for ing in parsed)


def migrate():
    call, label = make_llm_caller()
    print("\n  backend : {}".format(label))

    with open(RECIPES_JSON, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    db      = load_nutrition_db()
    total   = len(recipes)
    updated = 0

    for i, recipe in enumerate(recipes):
        title    = recipe.get("title", "Untitled")
        prefix   = "[{}/{}]".format(i + 1, total)

        # ── Parse ingredient strings ───────────────────────────────────────
        if needs_parse(recipe):
            strings = [s.strip() for s in recipe.get("ingredients", []) if s.strip()]
            if not strings:
                print("{} Skipping (no ingredients): {}".format(prefix, title))
            else:
                print("{} Parsing  : {}".format(prefix, title))
                parsed_map = parse_ingredient_strings(strings, call)
                recipe["ingredientsParsed"] = [
                    parsed_map.get(s, {"amount": "", "unit": "", "name": s})
                    for s in strings
                ]
                updated += 1
        else:
            print("{} Already parsed: {}".format(prefix, title))

        # ── Match to nutrition DB ──────────────────────────────────────────
        if needs_match(recipe):
            print("    Matching nutrition...")
            parsed, unmatched = match_ingredients(recipe["ingredientsParsed"], db, call)
            recipe["ingredientsParsed"] = parsed

            for ing in recipe["ingredientsParsed"]:
                name  = ing.get("name", "")
                nid   = ing.get("nutritionId")
                if nid:
                    print("    matched  : {}  ->  {} (#{})" .format(
                        name, ing.get("nutritionMatch", ""), nid))
                else:
                    print("    new entry: {}".format(name))

            if unmatched:
                print("    Generating {} new nutrition entr{}...".format(
                    len(unmatched), "y" if len(unmatched) == 1 else "ies"))
                name_to_id, db = create_missing_ingredients(unmatched, db, call)

                for ing in recipe["ingredientsParsed"]:
                    if ing.get("nutritionId") is None:
                        orig   = ing.get("name", "")
                        new_id = name_to_id.get(orig)
                        if new_id:
                            ing["nutritionId"]    = new_id
                            entry = next((e for e in db if e["id"] == new_id), {})
                            ing["nutritionMatch"] = entry.get("name", orig)
                            print("    created  : {}  ->  #{}".format(orig, new_id))

            updated += 1

        # ── Remove static nutrition fields ─────────────────────────────────
        for field in ("calories", "protein", "carbs"):
            recipe.pop(field, None)

    # Save
    with open(RECIPES_JSON, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)

    print("\nDone. Processed {}/{} recipes. recipes.json saved.\n".format(updated, total))


if __name__ == "__main__":
    migrate()
