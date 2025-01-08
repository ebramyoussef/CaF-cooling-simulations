import QuantumStates
import OpticalBlochEquations
import Distributed
import StaticArrays
import UnitsToValue: μB, gS, h
include("CaF_X.jl")
include("CaF_A.jl")
# Define constants for the laser cooling transition
Distributed.@everywhere begin
    QuantumStates.@consts begin
        λ = 606e-9
        Γ = 2π * 8.3e6
        m = UnitsToValue.@with_unit 59 "u"
        k = 2π / λ
    end
end

H = QuantumStates.CombinedHamiltonian([X_state_ham, A_state_ham])
QuantumStates.evaluate!(H)
QuantumStates.solve!(H)
QuantumStates.update_basis_tdms!(H)
QuantumStates.update_tdms!(H)

ground_state_idxs = 1:12
excited_state_idxs = 17:20
states_idxs = [ground_state_idxs; excited_state_idxs]

ground_states = H.states[ground_state_idxs]
excited_states = H.states[excited_state_idxs]

d = H.tdms[states_idxs, states_idxs, :]
states = H.states[states_idxs]

Zeeman_x(state, state′) = (QuantumStates.Zeeman(state, state′, -1) - QuantumStates.Zeeman(state, state′, 1)) / √2
Zeeman_y(state, state′) = im * (QuantumStates.Zeeman(state, state′, -1) + QuantumStates.Zeeman(state, state′, 1)) / √2
Zeeman_z(state, state′) = QuantumStates.Zeeman(state, state′, 0)

Zeeman_x_mat = real.(OpticalBlochEquations.operator_to_matrix(Zeeman_x, ground_states) .* (1e-4 * gS * μB * (2π / Γ) / h))
Zeeman_y_mat = imag.(OpticalBlochEquations.operator_to_matrix(Zeeman_y, ground_states) .* (1e-4 * gS * μB * (2π / Γ) / h))
Zeeman_z_mat = real.(OpticalBlochEquations.operator_to_matrix(Zeeman_z, ground_states) .* (1e-4 * gS * μB * (2π / Γ) / h))

# Stark_x(state, state′) = (Stark(state, state′,-1) - Zeeman(state, state′,1)) / √2

# Stark_x_mat = real.(operator_to_matrix(Stark_x, ground_states) .* (1e-4 * gS * μB * (2π/Γ) / h)) # factors???

@everywhere import LoopVectorization: @turbo
@everywhere function add_terms_dψ!(dψ, ψ, p, r, t)
    @turbo for i ∈ 1:12
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for j ∈ 1:12
            ψ_i_re = ψ.re[j]
            ψ_i_im = ψ.im[j]

            H_re = p.sim_params.Bx * p.sim_params.Zeeman_Hx[i, j] + p.sim_params.Bz * p.sim_params.Zeeman_Hz[i, j]
            H_im = p.sim_params.By * p.sim_params.Zeeman_Hy[i, j]

            dψ_i_re += ψ_i_re * H_re - ψ_i_im * H_im
            dψ_i_im += ψ_i_re * H_im + ψ_i_im * H_re

        end
        dψ.re[i] += dψ_i_im
        dψ.im[i] -= dψ_i_re
    end
    return nothing
end