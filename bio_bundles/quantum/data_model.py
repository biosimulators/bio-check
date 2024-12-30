import numpy as np


class DiracNotation(np.ndarray):
    def __new__(cls, values: list[complex]):
        return np.asarray(values).view(cls)


class Bra(DiracNotation):
    pass


class Ket(DiracNotation):
    def bra(self) -> Bra:
        ket_value = self.view()
        return Bra(np.conj(ket_value).T)

