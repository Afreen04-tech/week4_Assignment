import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

texts = {
    "Sentence": "Artificial intelligence helps businesses automate customer support.",

    "Paragraph": """Large language models are transforming the way businesses
    communicate with customers. They can answer questions, summarize documents,
    classify support tickets and generate useful responses. However, they can
    sometimes produce incorrect information. Therefore, companies must evaluate
    their accuracy, reliability and cost before using them in production.""",

    "Page": """Large language models are powerful artificial intelligence systems.
    They are trained on large amounts of text and other data. They can perform
    different tasks such as summarization, translation, classification and
    information extraction.

    Tokenization is the process of converting text into smaller units called
    tokens. A token can represent a word, part of a word or punctuation mark.
    Different languages may require different numbers of tokens for the same
    amount of text.

    Context windows determine how much information a model can process in one
    request. A larger context window helps models understand longer documents.
    However, larger requests may increase processing costs.

    Companies use language models for chatbots, document analysis, coding and
    customer support. Before selecting a model, organizations should compare
    accuracy, speed, pricing, privacy and context window limitations.""",

    "JSON": """{
        "ticket_id": 101,
        "customer": "Afreen",
        "issue": "Payment failed",
        "priority": "High",
        "status": "Open"
    }""",

    "Tamil": "வணக்கம்! செயற்கை நுண்ணறிவு வாடிக்கையாளர் சேவையை மேம்படுத்துகிறது."
}

print("TOKEN COUNT RESULTS")
print("-" * 75)

for name, text in texts.items():
    tokens = encoding.encode(text)
    words = len(text.split())
    token_count = len(tokens)
    ratio = words / token_count

    print(f"\nText: {name}")
    print(f"Words: {words}")
    print(f"Tokens: {token_count}")
    print(f"Words per token: {ratio:.2f}")
    print(f"Fits 8K: {'Yes' if token_count <= 8000 else 'No'}")
    print(f"Fits 128K: {'Yes' if token_count <= 128000 else 'No'}")


def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    input_cost = (input_tokens / 1_000_000) * price_in
    output_cost = (output_tokens / 1_000_000) * price_out
    return input_cost + output_cost


chats_per_day = 1000
input_per_chat = 1500
output_per_chat = 400

monthly_input = chats_per_day * input_per_chat * 30
monthly_output = chats_per_day * output_per_chat * 30

price_in = 0.25
price_out = 1.50

daily_cost = estimate_cost(
    1000 * 1500,
    1000 * 400,
    price_in,
    price_out
)

monthly_cost = estimate_cost(
    monthly_input,
    monthly_output,
    price_in,
    price_out
)

print("\nMONTHLY COST CALCULATION")
print("-" * 75)
print("Chats per day:", chats_per_day)
print("Input tokens per chat:", input_per_chat)
print("Output tokens per chat:", output_per_chat)
print("Monthly input tokens:", monthly_input)
print("Monthly output tokens:", monthly_output)
print(f"Daily cost: ${daily_cost:.2f}")
print(f"Monthly cost: ${monthly_cost:.2f}")