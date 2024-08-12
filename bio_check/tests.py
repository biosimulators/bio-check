from .verifier import Verifier


def test_verify_sbml():
    verifier = Verifier()
    assert verifier._test_root() is not None


def test_verify_omex():
    verifier = Verifier()
    assert verifier._test_root() is not None


