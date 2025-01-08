# Define a Hund's case (b) basis for the Hamiltonian, using states from N=0 to N=3
import QuantumStates
QN_bounds = (
    label="X",
    S=1 / 2,
    I=1 / 2,
    Λ=0,
    N=1
)
X_state_basis = QuantumStates.order_basis_by_m(QuantumStates.enumerate_states(QuantumStates.HundsCaseB_LinearMolecule, QN_bounds))

# Define the operator for the X state Hamiltonian of CaOH
X_state_operator = :(
    BX * QuantumStates.Rotation +
    DX * QuantumStates.RotationDistortion +
    γX * QuantumStates.SpinRotation +
    cX * (QuantumStates.Hyperfine_Dipolar / 3) +
    s * QuantumStates.basis_splitting
);

X_state_parameters = QuantumStates.@params begin
    BX = 10303.988 * 1e6
    DX = 0.014060 * 1e6
    γX = 39.65891 * 1e6
    bX = 109.1839 * 1e6
    cX = 40.1190 * 1e6
    s = 1
end

X_state_ham = QuantumStates.Hamiltonian(basis=X_state_basis, operator=X_state_operator, parameters=X_state_parameters)
QuantumStates.evaluate!(X_state_ham)
QuantumStates.solve!(X_state_ham)