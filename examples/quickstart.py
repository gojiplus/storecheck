"""End to end: lint the store, poison it, find the poison by influence, gate the change.

Run:  python examples/quickstart.py
"""

from storecheck import (
    Case,
    EmbeddingRetriever,
    EvalSet,
    HashingEmbedder,
    LabelDef,
    MajorityVoteDecider,
    Objective,
    Row,
    Store,
    Taxonomy,
    run_static,
)
from storecheck.checks import (
    DecisionLayerCheck,
    Evaluator,
    MutationCheck,
    RegressionCheck,
    row_influence,
)
from storecheck.checks.base import CheckContext
from storecheck.runner import run_checks

tax = Taxonomy(
    [
        LabelDef(
            "authentication_failure",
            "User could not prove identity: bad password, expired session, MFA.",
        ),
        LabelDef(
            "authorization_failure",
            "User proved identity but lacks permission for the resource.",
        ),
        LabelDef("billing", "Charges, invoices, refunds, payment methods."),
    ]
)


def row(text: str, label: str) -> Row:
    """Construct a demo row with its current taxonomy label version.

    Args:
        text: Example text.
        label: Reference label identifier.

    Returns:
        Construct a demo row with its current taxonomy label version.
    """
    return Row(
        text=text, label=label, author="demo", taxonomy_version=tax.label_version(label)
    )


store = Store(
    [
        row("password rejected when I try to log in", "authentication_failure"),
        row("login says my password is wrong", "authentication_failure"),
        row("my session expired and I had to sign in again", "authentication_failure"),
        row(
            "two factor code never arrives so I cannot log in", "authentication_failure"
        ),
        row(
            "I am logged in but the admin page says access denied",
            "authorization_failure",
        ),
        row(
            "signed in fine, still cannot open the reports folder, permission denied",
            "authorization_failure",
        ),
        row(
            "logged in as a viewer and cannot edit the document",
            "authorization_failure",
        ),
        row("I was charged twice for the same order", "billing"),
        row("need a refund for the duplicate invoice", "billing"),
        row("update the card on file for my subscription", "billing"),
    ],
    name="authz-demo",
)

evalset = EvalSet(
    [
        Case(
            "c1",
            "wrong password error on login",
            "authentication_failure",
            frozenset({"authn"}),
        ),
        Case(
            "c2",
            "it keeps logging me out and asking for the password",
            "authentication_failure",
            frozenset({"authn"}),
        ),
        Case(
            "c3",
            "the 2fa text message does not come so login fails",
            "authentication_failure",
            frozenset({"authn"}),
        ),
        Case(
            "c4",
            "logged in but access denied on the admin page",
            "authorization_failure",
            frozenset({"authz"}),
            challenge=True,
        ),
        Case(
            "c5",
            "signed in, permission denied opening the reports folder",
            "authorization_failure",
            frozenset({"authz"}),
            challenge=True,
        ),
        Case(
            "c6",
            "viewer role cannot edit the document even though logged in",
            "authorization_failure",
            frozenset({"authz"}),
        ),
        Case(
            "c7",
            "charged twice, want one charge refunded",
            "billing",
            frozenset({"billing"}),
        ),
        Case("c8", "please change the card on file", "billing", frozenset({"billing"})),
    ]
)

embedder = HashingEmbedder()
evaluator = Evaluator(EmbeddingRetriever(embedder, k=3), MajorityVoteDecider(), evalset)

print("== static tier ==")
print(run_static(store, tax, embedder).summary())

print("\n== baseline ==")
base = evaluator.run(store)
print(f"accuracy {base.metric():.3f}  by slice {base.by_slice()}")

print("\n== a proposed edit: someone adds a plausible-looking row ==")
poison = row(
    "logged in but access denied, permission denied on the admin "
    "page and reports folder",
    "authentication_failure",
)
candidate = store.add(poison)
objective = Objective(max_right_to_wrong=0, min_slice_metric={"authz": 0.9})
gate = run_checks(
    candidate,
    [RegressionCheck(evaluator, baseline=store, objective=objective)],
    tax,
    embedder,
)
print(gate.summary())
print("merge allowed:", gate.passed())

print(
    "\n== row influence on the poisoned store (leave-one-out, context-diff shortcut) =="
)
calls0 = evaluator.decider.calls
for inf in row_influence(evaluator, candidate)[:3]:
    print(
        f"  {inf.row.id} net={inf.net:+d} fixes={inf.hurts} "
        f"breaks={inf.helps} retrieved_for={inf.retrieved_for}  "
        f"[{inf.row.label}] {inf.row.text[:50]}"
    )
print(f"  decider calls for influence pass: {evaluator.decider.calls - calls0}")

print("\n== decision layer vs store ==")
for f in DecisionLayerCheck(evaluator).run(CheckContext(store)):
    print("  ", f.message)

print("\n== meta: does the evaluator notice deliberate damage? ==")
for f in MutationCheck(evaluator).run(CheckContext(store)):
    print(f"  {f.severity.name:5} {f.message}")
