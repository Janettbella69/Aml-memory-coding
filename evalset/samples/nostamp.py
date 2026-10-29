"""Sample 'nostamp': short bakery-planning chat where the adapter sends NO timestamps; dates live in the text.

Checks that the service degrades gracefully without `timestamp` (ordering by ordinal, recency fallback).
"""
from evalset.schema import Q, Sample, session

SESSIONS = (
    session(None, [
        ("user", "Today is 3 February 2026 and I've decided to open a small bakery in Pune. I'm Ravi, I've been a pastry chef for eight years.", "ravi-intro"),
        ("assistant", "Congratulations on the decision, Ravi. What's the first milestone?"),
        ("user", "Finding a shop. Budget for the lease is 60,000 rupees a month, maximum.", "lease-budget"),
        ("assistant", "That's a sensible cap for a small unit in Pune. Which areas?"),
        ("user", "Koregaon Park or Kalyani Nagar. Nothing else."),
        ("assistant", "Both have good footfall. Shall we list what to check in each shop?"),
        ("user", "Later. One rule: I'm vegetarian and the bakery will be fully eggless, so never suggest recipes with eggs.", "rule-eggless"),
        ("assistant", "Understood, eggless only, always."),
    ]),
    session(None, [
        ("user", "It's 21 February 2026. I signed the lease today, Kalyani Nagar, 52,000 rupees a month, three-year term. Opening target is 15 April.", "lease-signed"),
        ("assistant", "Under budget and a three-year term, well done. Seven and a half weeks to opening."),
        ("user", "The oven arrives on 10 March. It's a Rational deck oven, 3 decks, bought second-hand for 2.4 lakh.", "oven"),
        ("assistant", "A second-hand Rational is a good buy. Shall I draft an opening countdown?"),
        ("user", "Yes. And the shop name is Crumb & Cardamom.", "shop-name"),
        ("assistant", "Lovely name. Countdown: 10 March oven, 20 March trial bakes, 1 April soft opening for friends, 15 April public opening."),
    ]),
    session(None, [
        ("user", "Update on 28 March 2026: the oven arrived late, on 18 March, so trial bakes only started on the 22nd. I'm pushing the public opening to 25 April.", "opening-moved"),
        ("assistant", "A ten-day slip is manageable. Keep the soft opening on 1 April?"),
        ("user", "Soft opening moves to 10 April. Also my sister Meera is joining as co-owner, she's putting in 5 lakh.", "meera"),
        ("assistant", "Welcome Meera. Shall we talk about how to split responsibilities?"),
        ("user", "She does front of house and accounts, I bake. That's the whole plan."),
        ("assistant", "Simple and clear."),
    ]),
)

QUESTIONS = (
    Q("ns-te-01", "temporal/dates_intervals", "When does Crumb & Cardamom open to the public?", ["opening-moved"], trap=["lease-signed"], answer="25 April 2026"),
    Q("ns-fr-01", "fact_recall/single_fact", "How much is Ravi's monthly lease?", ["lease-signed"], trap=["lease-budget"], answer="52,000 rupees"),
    Q("ns-te-02", "temporal/event_ordering", "Did the oven arrive before or after the lease was signed?", ["lease-signed", "opening-moved"], mode="all", answer="after (lease 21 Feb, oven 18 Mar)"),
    Q("ns-ex-01", "execution/rule_application", "Ravi asks for a brioche recipe. What constraint applies?", ["rule-eggless"], answer="it must be eggless"),
    Q("ns-fr-02", "fact_recall/entity_attribution", "Who is Meera and what is her role in the bakery?", ["meera"], answer="Ravi's sister, co-owner, front of house and accounts"),
    Q("ns-sa-01", "safety/abstention", "How many employees has Ravi hired for the bakery?", [], answer="(never mentioned)"),
)

SAMPLE = Sample(name="nostamp", style="assistant", sessions=SESSIONS, questions=QUESTIONS,
                description="Bakery planning, 3 sessions, no timestamps sent; dates only inside the text.")
