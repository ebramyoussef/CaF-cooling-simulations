QN_bounds = (
    label="A",
    S=1 / 2,
    I=1 / 2,
    Λ=(-1, 1),
    J=1 / 2
)
A_state_basis = order_basis_by_m(enumerate_states(HundsCaseA_LinearMolecule, QN_bounds))

A_state_operator = :(
    T_A * DiagonalOperator +
    Be_A * Rotation +
    Aso_A * SpinOrbit +
    q_A * (ΛDoubling_q + 2ΛDoubling_p2q) +
    p_A * ΛDoubling_p2q +
    # B_z * Zeeman_L +
    # b00_A * Hyperfine_IL +
    # b00_A * Hyperfine_IF +
    b00_A * (Hyperfine_IF - Hyperfine_IL) +
    s * basis_splitting
)

# Spectroscopic constants for CaOH, A state
A_state_parameters = QuantumStates.@params begin
    T_A = 16526.750 * c * 1e2
    Be_A = 0.348781 * c * 1e2
    Aso_A = 71.429 * c * 1e2
    b00_A = 1e6
    p_A = -0.044517 * c * 1e2
    q_A = -2.916e-4 * c * 1e2
    B_z = 0.0
    s = 1e6
end;

A_state_ham_caseA = Hamiltonian(basis=A_state_basis, operator=A_state_operator, parameters=A_state_parameters)
evaluate!(A_state_ham_caseA)
QuantumStates.solve!(A_state_ham_caseA)

# Convert A state from Hund's case (a) to Hund's case (b)
QN_bounds = (
    label="A",
    S=1 / 2,
    I=1 / 2,
    Λ=(-1, 1),
    N=1,
    J=1 / 2
)
A_state_caseB_basis = order_basis_by_m(enumerate_states(HundsCaseB_LinearMolecule, QN_bounds))
A_state_ham = convert_basis(A_state_ham_caseA, A_state_caseB_basis)