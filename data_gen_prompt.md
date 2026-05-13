You are generating a fine-tuning dataset for a small language model that converts voice-transcribed transaction descriptions into structured JSON. The model will run on Android devices to help users log expenses by speaking.

## TASK
Generate 150 unique training examples in JSONL format. Each line must be a valid JSON object with exactly two keys: "input" and "output".

## OUTPUT SCHEMA (strict)
The "output" field must be a JSON object with this exact structure:
{
  "transactions": [
    {
      "amount": <number>,
      "currency": "INR",
      "item": "<string, lowercase, singular noun phrase>",
      "category": "<one of: Food, Drinks, Groceries, Transport, Shopping, Entertainment, Bills, Health, Education, Personal, Gifts, Income, Other>",
      "type": "<expense | income>"
    }
  ]
}

## INPUT CHARACTERISTICS
Inputs are voice transcripts in English, but reflect how Indian users actually speak. Include this variety across the 150 examples:

1. **Clean formal** (10%): "I spent 500 rupees on beer and 50 rupees on candy"
2. **Casual short** (25%): "500 rs beer 50 rs candy", "200 uber", "1k groceries"
3. **Hinglish mixed** (20%): "do sau rupay ka chai", "paanch sau on dinner", "char hazaar rent"
4. **Numbers as words** (10%): "five hundred on petrol", "two thousand for medicines"
5. **Abbreviated amounts** (10%): "1.5k for shoes", "2k uber", "500/- chai", "₹300 on lunch"
6. **Multiple transactions** (15%): "300 lunch 50 chai 200 uber back home"
7. **Income mentions** (5%): "got 5000 salary", "received 200 cashback from paytm"
8. **Corrections/disfluency** (3%): "500 on beer wait no 600 on beer", "umm 200 for uber"
9. **Ambiguous / requires inference** (2%): "paid 500 for that thing", "spent 300 yesterday"

## RULES
- Amounts: parse "k" as ×1000, "lakh" as ×100000, "hazaar" as ×1000, "sau" as ×100. Numbers-as-words must be converted to digits in output.
- Currency: always "INR" since users are Indian. Treat "$", "dollars" only if explicitly mentioned (then use "USD").
- Item field: normalize to lowercase singular noun ("beer" not "Beers", "uber ride" not "Uber").
- Category: pick the BEST fit. "uber/ola/auto/petrol/bus" → Transport. "beer/wine/chai/coffee" → Drinks. "rent/electricity/wifi/recharge" → Bills. "movie/netflix/concert" → Entertainment. "doctor/medicine/hospital" → Health.
- Type: "expense" by default. Only "income" for explicit salary, cashback, refund, gift received, payment received.
- For ambiguous inputs ("that thing", "stuff"), use item: "unspecified" and category: "Other".
- For disfluency ("wait no 600"), use the CORRECTED amount only.
- Each input must be unique. Vary sentence structure, item types, amounts (use a wide range: 10 to 200000), and phrasing.

## EXAMPLES (do not repeat these, generate 150 NEW ones)

{"input": "500 rs on beer 50 rs on candy", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}, {"amount": 50, "currency": "INR", "item": "candy", "category": "Food", "type": "expense"}]}}
{"input": "do sau rupay ka chai", "output": {"transactions": [{"amount": 200, "currency": "INR", "item": "chai", "category": "Drinks", "type": "expense"}]}}
{"input": "got 5000 salary today", "output": {"transactions": [{"amount": 5000, "currency": "INR", "item": "salary", "category": "Income", "type": "income"}]}}
{"input": "1.5k for shoes from myntra", "output": {"transactions": [{"amount": 1500, "currency": "INR", "item": "shoes", "category": "Shopping", "type": "expense"}]}}
{"input": "300 lunch 50 chai 200 uber back home", "output": {"transactions": [{"amount": 300, "currency": "INR", "item": "lunch", "category": "Food", "type": "expense"}, {"amount": 50, "currency": "INR", "item": "chai", "category": "Drinks", "type": "expense"}, {"amount": 200, "currency": "INR", "item": "uber ride", "category": "Transport", "type": "expense"}]}}
{"input": "500 on beer wait no 600 on beer", "output": {"transactions": [{"amount": 600, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}]}}

## BATCH_FOCUS
[CHANGE THIS LINE EACH BATCH — e.g., "Focus this batch on Hinglish inputs and Bills category" or "Focus on multi-transaction inputs with 3-5 items each" or "Focus on edge cases: corrections, ambiguity, very large or very small amounts"]

## OUTPUT FORMAT
Output ONLY 150 JSONL lines. No preamble, no explanation, no markdown fences. One JSON object per line.