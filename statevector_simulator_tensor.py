#!/usr/bin/env python3

"""
This module is made to improve the speed of statevector simulations for quantum
circuits in python using tensors.

Author: Floyd Creevey
Email: fc309@sanger.ac.uk
"""

import numpy as np

H_gate = (1 / np.sqrt(2)) * np.array([[1 + 0j, 1 + 0j], [1 + 0j, -1 + 0j]])

X_gate = np.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])

cx_gate = np.reshape(
    np.array(
        [
            [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
            [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
        ]
    ),
    (2, 2, 2, 2),
)


def rotation_x(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2) + 0j, 0 + -1j * np.sin(theta / 2)],
            [0 + -1j * np.sin(theta / 2), np.cos(theta / 2) + 0j],
        ]
    )


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2) + 0j, -np.sin(theta / 2) + 0j],
            [np.sin(theta / 2) + 0j, np.cos(theta / 2) + 0j],
        ]
    )


def rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [0 + np.e ** (-1j * (theta / 2)), 0 + 0j],
            [0 + 0j, 0 + np.e ** (1j * (theta / 2))],
        ]
    )


def controlled_rotation_x(theta: float) -> np.ndarray:
    return np.reshape(
        np.array(
            [
                [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
                [0 + 0j, 0 + 0j, np.cos(theta / 2) + 0j, 0 + -1j * np.sin(theta / 2)],
                [0 + 0j, 0 + 0j, 0 + -1j * np.sin(theta / 2), np.cos(theta / 2) + 0j],
            ]
        ),
        (2, 2, 2, 2),
    )


def controlled_rotation_y(theta: float) -> np.ndarray:
    return np.reshape(
        np.array(
            [
                [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
                [0 + 0j, 0 + 0j, np.cos(theta / 2) + 0j, -np.sin(theta / 2) + 0j],
                [0 + 0j, 0 + 0j, np.sin(theta / 2) + 0j, np.cos(theta / 2) + 0j],
            ]
        ),
        (2, 2, 2, 2),
    )


def controlled_rotation_z(theta: float) -> np.ndarray:
    return np.reshape(
        np.array(
            [
                [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
                [0 + 0j, 0 + 0j, 0 + np.exp(-1j * theta / 2), 0 + 0j],
                [0 + 0j, 0 + 0j, 0 + 0j, np.exp(1j * theta / 2) + 0j],
            ]
        ),
        (2, 2, 2, 2),
    )


class Reg:
    def __init__(self, n: int) -> None:
        self.n = n
        self.psi = np.zeros((2,) * n, dtype=np.complex128)
        self.psi[(0,) * n] = 1

    def h(self, i: int) -> None:
        self.psi = np.tensordot(H_gate, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def x(self, i: int) -> None:
        self.psi = np.tensordot(X_gate, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def rx(self, theta: float, i: int) -> None:
        self.psi = np.tensordot(rotation_x(theta), self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def ry(self, theta: float, i: int) -> None:
        self.psi = np.tensordot(rotation_y(theta), self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def rz(self, theta: float, i: int) -> None:
        self.psi = np.tensordot(rotation_z(theta), self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def cx(self, theta: float, i: int) -> None:
        self.psi = np.tensordot(cx_gate, self.psi, ((2, 3), (control, target)))
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))

    def crx(self, theta: float, i: int) -> None:
        self.psi = np.tensordot(
            controlled_rotation_x(theta), self.psi, ((2, 3), (control, target))
        )
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))

    def cry(self, theta: float, control: int, target: int) -> None:
        self.psi = np.tensordot(
            controlled_rotation_y(theta), self.psi, ((2, 3), (control, target))
        )
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))

    def crz(self, theta: float, control: int, target: int) -> None:
        self.psi = np.tensordot(
            controlled_rotation_z(theta), self.psi, ((2, 3), (control, target))
        )
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))

    def unitary(self, unitary: np.ndarray, qubits: list[int, int]) -> None:
        if len(qubits) == 2:
            self.psi = np.tensordot(
                unitary.reshape(2, 2, 2, 2), self.psi, ((2, 3), (qubits[0], qubits[1]))
            )
            self.psi = np.moveaxis(self.psi, (0, 1), (qubits[0], qubits[1]))
        else:
            self.psi = np.tensordot(unitary, self.psi, (1, qubits[0]))
            self.psi = np.moveaxis(self.psi, 0, qubits[0])