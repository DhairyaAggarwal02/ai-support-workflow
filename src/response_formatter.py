import re


SECTION_SPLIT_PATTERN = r"(?=##\s+)"


def clean_text(text):
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sections(chunk_text):
    raw_sections = re.split(SECTION_SPLIT_PATTERN, chunk_text)

    sections = []

    for section in raw_sections:
        cleaned = clean_text(section)

        # Skip document-level titles like "Shipping Policy"
        if cleaned.lower() in [
            "shipping policy",
            "returns and refunds policy",
            "payments policy",
            "subscription policy",
            "saas billing policy",
            "account access policy",
            "technical issues policy",
            "saas subscription policy",
            "escalation rules",
            "saas escalation rules",
        ]:
            continue

        if len(cleaned.split()) >= 6:
            sections.append(cleaned)

    return sections


def score_section(query, section):
    query_words = [
        word.lower()
        for word in re.findall(r"\b\w+\b", query)
        if len(word) > 3
    ]

    section_lower = section.lower()

    return sum(word in section_lower for word in query_words)


def format_response(query, chunk_text, source_file):
    sections = get_sections(chunk_text)

    if not sections:
        fallback = clean_text(chunk_text)
        return f"{fallback}\n\n(Source: {source_file})"

    scored = [
        (score_section(query, section), section)
        for section in sections
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_section = scored[0]

    # Remove heading-like first phrase if present
    best_section = re.sub(
        r"^(Standard Shipping|Express Shipping|Order Processing|Lost Packages|International Shipping|Invoice Requests|Password Reset|Failed Payments|Duplicate Charges|Accepted Payment Methods|Login Problems|API Errors|Plan Changes|Cancellation|Renewal|Refunds)\s+",
        "",
        best_section
    )

    return f"{best_section}\n\n(Source: {source_file})"