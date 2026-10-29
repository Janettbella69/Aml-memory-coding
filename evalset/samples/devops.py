"""Sample 'devops': a backend lead and a coding assistant; rule- and workflow-heavy (assistant style).

Fictional company "Ledgerly". Exercises Execution (rules, workflow, format constraints, empirical data,
domain reasoning) and Governance (a rule that is later replaced).
"""
from evalset.schema import Q, Sample, session

SESSIONS = (
    session("2025-05-05 09:00", [
        ("user", "I'm Sam, backend lead on an app called Ledgerly, an invoicing tool for freelancers. Stack: TypeScript, Node 22, Fastify, Postgres 16 on RDS. Monorepo.", "stack"),
        ("assistant", "Thanks Sam. What do you want help with first?"),
        ("user", "Ground rules. Always use pnpm for package commands in this repo, never npm or yarn. Our lockfile is pnpm-lock.yaml and CI fails on anything else.", "rule-pnpm"),
        ("assistant", "Understood: pnpm only, never npm or yarn, in the Ledgerly repo."),
        ("user", "Second: commit messages follow Conventional Commits. feat:, fix:, chore:, docs:, with a scope in parentheses when it's obvious, like fix(invoices): handle zero totals.", "rule-commits"),
        ("assistant", "Got it. Conventional Commits with an optional scope."),
        ("user", 'Third: all API errors must be returned as JSON in the shape { "error": { "code": "string", "message": "string" } }. No bare strings, no stack traces.', "rule-error-format"),
        ("assistant", "Noted: every API error is { error: { code, message } }, nothing else."),
        ("user", "Good. We have around 2,000 active users and the DB connection limit on the RDS instance is 200.", "db-limit"),
        ("assistant", "Two thousand users and a 200-connection cap. We should talk about pooling when we get to deployment."),
    ]),
    session("2025-05-19 15:30", [
        ("user", "Our release workflow, so you can remind me: 1) bump the version in package.json, 2) run pnpm test, 3) tag vX.Y.Z, 4) push the tag, 5) the GitHub Action builds the image and deploys to Fly.io. Never deploy from a laptop.", "workflow-release"),
        ("assistant", "Captured: bump version, pnpm test, tag vX.Y.Z, push tag, GitHub Action deploys to Fly.io. Never from a laptop."),
        ("user", "We run three app instances on Fly, all sharing that single RDS database.", "instances-3"),
        ("assistant", "Three instances against a 200-connection limit: each pool should stay well under 60 so you leave headroom for migrations and admin sessions."),
        ("user", "That matches what we do, pool size 40 per instance."),
        ("assistant", "Then you're at 120 of 200 at peak, which is comfortable."),
        ("user", "Side project note, unrelated to Ledgerly: I keep a tiny personal blog repo that still uses npm because it's a ten-year-old Jekyll thing with a package.json I never touched. Don't apply the pnpm rule there.", "side-npm"),
        ("assistant", "Understood: the pnpm rule is for Ledgerly only; the blog stays on npm."),
    ]),
    session("2025-06-09 10:15", [
        ("user", "Performance work. Before any caching, the p95 latency on GET /invoices was 420 ms, measured over a week in Grafana.", "p95-before"),
        ("assistant", "420 ms p95 for a list endpoint is slow. Where does the time go?"),
        ("user", "Mostly the totals query. We added a Redis cache for invoice totals with a 60 second TTL. After the cache, p95 dropped to 180 ms.", "p95-after"),
        ("assistant", "From 420 to 180 ms is a 57% reduction. Nice. Is Redis on Fly as well?"),
        ("user", "Upstash Redis, in the same region. One caveat: we invalidate on invoice update, so stale totals are at most a few seconds in practice."),
        ("assistant", "That's a sound design."),
        ("user", "Also a format rule for you: when I ask for SQL, use lowercase keywords and put each clause on its own line. I find uppercase SQL shouty.", "rule-sql-format"),
        ("assistant", "Noted: lowercase keywords, one clause per line, whenever I write SQL for you."),
    ]),
    session("2025-06-30 14:00", [
        ("user", "Incident today. A deploy took the API down for 11 minutes because a migration locked the invoices table. Root cause: adding a NOT NULL column with a volatile default on Postgres rewrote the table.", "incident-migration"),
        ("assistant", "On Postgres 16, adding a column with a constant default is fast, but a volatile default such as now(), or a backfill in the same transaction, rewrites or locks the table. What was the default?"),
        ("user", "now(). Volatile. So yes. New rule: migrations that touch the invoices table must be run in two steps, add the nullable column first, backfill in batches, then add the constraint.", "rule-migration"),
        ("assistant", "Captured: two-step migrations on the invoices table: nullable column, batched backfill, then the constraint."),
        ("user", "Postmortem is written. We also added a step 2b to the release workflow: run pnpm migrate:dry against a staging snapshot before tagging.", "workflow-2b"),
        ("assistant", "Updated workflow: bump, test, migrate dry-run on a staging snapshot, tag, push, deploy."),
    ]),
    session("2025-07-21 09:40", [
        ("user", "Hiring news: we got a second backend engineer, Priya, who starts 4 August. She'll own the payments integration.", "priya"),
        ("assistant", "Welcome Priya. Shall I keep a note of what she owns?"),
        ("user", "Yes, payments is hers. I keep invoices and auth."),
        ("assistant", "Noted."),
        ("user", "And a preference for code reviews: when you review my diffs, lead with the one thing that would block merging, then nitpicks at the end, clearly labelled as nits.", "pref-review"),
        ("assistant", "Blocker first, nits last and labelled. Understood."),
    ]),
    session("2025-08-04 11:00", [
        ("user", "Big change. As of today, 4 August 2025, Ledgerly moved from pnpm to bun. Package commands are bun install, bun test, bun run. pnpm is gone from CI. Forget the pnpm rule for Ledgerly.", "rule-bun"),
        ("assistant", "Updated: Ledgerly uses bun from 4 August 2025; bun install, bun test, bun run. The old pnpm rule is retired for Ledgerly."),
        ("user", "Right. The release workflow step 2 is now bun test, and 2b is bun run migrate:dry.", "workflow-bun"),
        ("assistant", "Workflow is now: bump version, bun test, bun run migrate:dry against staging, tag vX.Y.Z, push tag, GitHub Action deploys to Fly.io."),
        ("user", "Exactly. Priya started today and she's already found two bugs in payments, which is a good sign."),
        ("assistant", "A productive first day."),
    ]),
    session("2025-08-25 16:20", [
        ("user", "Capacity planning. We're at 3,500 active users now and we're adding a fourth app instance next week.", "users-3500"),
        ("assistant", "With four instances at pool size 40 you'd be at 160 of 200 connections at peak. That leaves 40 for migrations and admin, which is thin but workable. Consider dropping the pool to 35."),
        ("user", "Let's do 35 per instance then. Also p95 on GET /invoices is holding at 190 ms with the extra load.", "pool-35"),
        ("assistant", "Four instances at 35 is 140 connections, more headroom. 190 ms is in line with the post-cache number."),
        ("user", "One more rule: never suggest a dependency with fewer than 1,000 weekly downloads without flagging it as risky.", "rule-deps"),
        ("assistant", "Noted: obscure dependencies always get flagged."),
    ]),
    session("2025-09-15 10:30", [
        ("user", "Postmortem follow-up: the two-step migration rule has been used four times since June without incident."),
        ("assistant", "That's the rule paying for itself."),
        ("user", 'We also changed the API error format slightly: errors now include an optional "request_id" field inside the error object, so it is { "error": { "code", "message", "request_id"? } }. Code and message are still required.', "error-format-v2"),
        ("assistant", "Updated: error objects may carry an optional request_id alongside the required code and message."),
        ("user", "And Priya shipped the Stripe integration on 10 September. Payments are live.", "stripe-live"),
        ("assistant", "Congratulations to Priya and the team."),
    ]),
)

QUESTIONS = (
    Q("dv-ex-01", "execution/rule_application", "Sam asks you to install a package in the Ledgerly repo. Which command do you use?", ["rule-bun"], trap=["rule-pnpm"], answer="bun install"),
    Q("dv-ex-02", "execution/workflow", "List the steps of Ledgerly's release workflow as of September 2025.", ["workflow-release", "workflow-2b", "workflow-bun"], mode="all",
      answer="bump version; bun test; bun run migrate:dry on staging snapshot; tag vX.Y.Z; push tag; GitHub Action deploys to Fly.io"),
    Q("dv-ex-03", "execution/format_constraints", "What JSON shape must Ledgerly API errors have?", ["error-format-v2", "rule-error-format"], answer="{ error: { code, message, request_id? } }"),
    Q("dv-ex-04", "execution/empirical_data", "What was the p95 latency of GET /invoices before and after the Redis cache?", ["p95-before", "p95-after"], mode="all", answer="420 ms before, 180 ms after"),
    Q("dv-ex-05", "execution/domain_reasoning", "With four instances and a pool of 35 each, how many RDS connections remain for migrations and admin?", ["pool-35", "db-limit"], mode="all", answer="60 (200 - 140)"),
    Q("dv-ex-06", "execution/format_constraints", "How should SQL be formatted when written for Sam?", ["rule-sql-format"], answer="lowercase keywords, one clause per line"),
    Q("dv-go-01", "governance/new_value_overwrite", "Which package manager does Ledgerly use now?", ["rule-bun"], trap=["rule-pnpm"], answer="bun (since 4 August 2025)"),
    Q("dv-go-02", "governance/contradiction", "Does Sam's personal blog use pnpm?", ["side-npm"], answer="no, it stays on npm; the pnpm rule was Ledgerly-only"),
    Q("dv-co-01", "composition/causal_chain", "Why was the two-step migration rule introduced?", ["incident-migration", "rule-migration"], answer="an 11-minute outage caused by a table-locking migration"),
    Q("dv-te-01", "temporal/dates_intervals", "When did Ledgerly switch to bun?", ["rule-bun"], answer="4 August 2025"),
    Q("dv-te-02", "temporal/event_ordering", "Did Priya join Ledgerly before or after the switch to bun?", ["priya", "rule-bun"], mode="all", answer="same day: both 4 August 2025"),
    Q("dv-fr-01", "fact_recall/single_fact", "Who owns the payments integration at Ledgerly?", ["priya"], answer="Priya"),
    Q("dv-fr-02", "fact_recall/structured_set", "What is Ledgerly's technology stack?", ["stack"],
      options=["A. Python, Django, MySQL", "B. TypeScript, Node 22, Fastify, Postgres 16", "C. Go, Gin, SQLite", "D. Ruby on Rails, Postgres"], answer="B"),
    Q("dv-pe-01", "personalization/preference_recall", "How does Sam want code reviews structured?", ["pref-review"], answer="blocker first, nits last and labelled"),
    Q("dv-co-02", "composition/multi_hop", "Which payment provider did the engineer who joined in August integrate?", ["priya", "stripe-live"], mode="all", answer="Stripe (Priya)"),
    Q("dv-sa-01", "safety/abstention", "What is Ledgerly's monthly revenue?", [], answer="(never mentioned)"),
    # v0.4 addition (2026-10-08): multiple-choice variant with an in-conversation distractor
    Q("dv-fr-03", "fact_recall/single_fact", "Who owns the payments integration at Ledgerly?", ["priya"], options=["A. Sam", "B. Priya", "C. Ravi", "D. Meera"], answer="B"),
)

SAMPLE = Sample(name="devops", style="assistant", sessions=SESSIONS, questions=QUESTIONS,
                description="Backend lead with a coding assistant, 8 sessions May-Sep 2025; standing rules, workflow edits, a rule replaced by another.")
