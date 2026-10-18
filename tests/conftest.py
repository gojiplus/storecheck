import pytest

from storecheck import (
    Case,
    EmbeddingRetriever,
    EvalSet,
    HashingEmbedder,
    LabelDef,
    MajorityVoteDecider,
    Row,
    Store,
    Taxonomy,
)
from storecheck.checks import Evaluator


@pytest.fixture
def taxonomy():
    return Taxonomy(
        [
            LabelDef(
                "authentication_failure",
                "The user could not prove who they are: bad "
                "password, expired session, MFA failure.",
            ),
            LabelDef(
                "authorization_failure",
                "The user proved who they are but lacks permission for the resource.",
            ),
            LabelDef("billing", "Charges, invoices, refunds, payment methods."),
        ]
    )


def _row(text, label, tax, **kw):
    kw.setdefault("author", "gaurav")
    kw.setdefault("taxonomy_version", tax.label_version(label))
    return Row(text=text, label=label, **kw)


@pytest.fixture
def store(taxonomy):
    t = taxonomy
    rows = [
        _row("password rejected when I try to log in", "authentication_failure", t),
        _row("login says my password is wrong", "authentication_failure", t),
        _row(
            "my session expired and I had to sign in again", "authentication_failure", t
        ),
        _row(
            "two factor code never arrives so I cannot log in",
            "authentication_failure",
            t,
        ),
        _row(
            "I am logged in but the admin page says access denied",
            "authorization_failure",
            t,
        ),
        _row(
            "signed in fine, still cannot open the reports folder, permission denied",
            "authorization_failure",
            t,
        ),
        _row(
            "logged in as a viewer and cannot edit the document",
            "authorization_failure",
            t,
        ),
        _row("I was charged twice for the same order", "billing", t),
        _row("need a refund for the duplicate invoice", "billing", t),
        _row("update the card on file for my subscription", "billing", t),
    ]
    return Store(rows, name="authz-demo")


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=512)


@pytest.fixture
def evalset():
    return EvalSet(
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
            Case(
                "c8",
                "please change the card on file",
                "billing",
                frozenset({"billing"}),
            ),
        ]
    )


@pytest.fixture
def evaluator(embedder, evalset):
    return Evaluator(EmbeddingRetriever(embedder, k=3), MajorityVoteDecider(), evalset)
