using Distributed
@everywhere using QuantumStates, OpticalBlochEquations, StaticArrays, StructArrays, DifferentialEquations
@everywhere using LoopVectorization: @turbo
@everywhere using UnitsToValue: μB, gS, h, c, ħ, kB
@everywhere using MutableNamedTuples: MutableNamedTuple

include("CaF_X.jl")
include("CaF_A.jl")

# Define constants for the laser cooling transition
@everywhere begin
	QuantumStates.@consts begin
		λ = 606e-9
		Γ = 2π * 8.3e6
		m = UnitsToValue.@with_unit 59 "u"
		k = 2π / λ
	end
end

# Define and evaluate Hamiltonian
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

energy_offset = (2π / Γ) * QuantumStates.energy(states[13]) #?
energies = QuantumStates.energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = +24
δ1 = +3

Δ1 = 1e6 * (detuning)
Δ2 = 1e6 * (detuning + δ1)

f1 = QuantumStates.energy(states[end]) - QuantumStates.energy(states[1]) + Δ1
f2 = QuantumStates.energy(states[end]) - QuantumStates.energy(states[12]) + Δ2

freqs = [f1, f2] .* (2π / Γ)

# DEFINE SATURATION INTENSITIES #
beam_radius = 5e-3
Isat = π * h * c * Γ / (3 * λ^3)
P = 2.5e-3 # 13.1 mW/1 V at 1.0 V, factor of 0.55 to match scattering rates
I = 2 * P / (π * beam_radius^2)

total_sat = I / Isat
s1 = total_sat/2
s2 = total_sat/2

sats = [s1, s2]

# DEFINE POLARIZATIONS #
pols = [OpticalBlochEquations.σ⁻, OpticalBlochEquations.σ⁺]

# DEFINE FUNCTION TO UPDATE PARAMETERS DURING SIMULATION #
@everywhere function update_p_SFcooling!(p, r, t)
	# s = p.sim_params.total_sat/2
	# s_factor = p.sim_params.s_factor_end
	# p.sats[1] = s * s_factor
	# p.sats[2] = s * s_factor
	return nothing
end

@everywhere function update_p_SFcooling_diffusion!(p, r, t)
	# s = p.sim_params.total_sat
	# s_factor = p.sim_params.s_factor_end
	# p.sats[1] = s * s_factor
	return nothing
end

# Define simulation parameters #

sim_type = Float64

σx_initial = 585e-6
σy_initial = 585e-6
σz_initial = 435e-6
Tx_initial = 35e-6
Ty_initial = 35e-6
Tz_initial = 35e-6


sim_params = MutableNamedTuple(
	Zeeman_Hx = QuantumStates.MMatrix{size(Zeeman_x_mat)...}(sim_type.(Zeeman_x_mat)),
	Zeeman_Hy = QuantumStates.MMatrix{size(Zeeman_y_mat)...}(sim_type.(Zeeman_y_mat)),
	Zeeman_Hz = QuantumStates.MMatrix{size(Zeeman_z_mat)...}(sim_type.(Zeeman_z_mat)), B_ramp_time = 4e-3 / (1 / Γ),
	B_grad_start = 0.0,
	B_grad_end = 74.0, s_ramp_time = 4e-3 / (1 / Γ),
	s_factor_start = 0.9,
	s_factor_end = 0.7, photon_budget = rand(Distributions.Geometric(1 / 13500)), x_dist = Distributions.Normal(0, σx_initial),
	y_dist = Distributions.Normal(0, σy_initial),
	z_dist = Distributions.Normal(0, σz_initial), vx_dist = Distributions.Normal(0, sqrt(kB * Tx_initial / 2m)),
	vy_dist = Distributions.Normal(0, sqrt(kB * Ty_initial / 2m)),
	vz_dist = Distributions.Normal(0, sqrt(kB * Tz_initial / 2m)), f_z = StructArrays.StructArray(zeros(Complex{sim_type}, 16, 16)),
	total_sat = 0.0, Bx = 0.0,
	By = 0.0,
	Bz = 0.0,
)

# PROBLEM TO CALCULATE TRAJECTORIES #

t_start = 0.0
t_end   = 1e-3
t_span  = (t_start, t_end) ./ (1 / Γ)

p_SFcooling = OpticalBlochEquations.initialize_prob(sim_type, energies, freqs, sats, pols, beam_radius, d, m / (ħ * k^2 / Γ), Γ, k, sim_params, update_p_SFcooling!, add_terms_dψ!)

cb1 = DifferentialEquations.ContinuousCallback(OpticalBlochEquations.condition_new, OpticalBlochEquations.stochastic_collapse_new!, save_positions = (false, false))
cb2 = DifferentialEquations.DiscreteCallback(terminate_condition, DifferentialEquations.terminate!)
cbs = DifferentialEquations.CallbackSet(cb1, cb2)

kwargs = (alg = DifferentialEquations.DP5(), reltol = 1e-4, saveat = 1000, maxiters = 200000000, callback = cbs)
prob_SFcooling = DifferentialEquations.ODEProblem(OpticalBlochEquations.ψ_fast!, p_SFcooling.u0, sim_type.(t_span), p_SFcooling; kwargs...)

# PROBLEM TO COMPUTE DIFFUSION #
p_SFcooling_diffusion = OpticalBlochEquations.initialize_prob(sim_type, energies, freqs, sats, pols, Inf, d, m / (ħ * k^2 / Γ), Γ, k, sim_params, update_p_SFcooling_diffusion!, add_terms_dψ!)

cb1_diffusion = DifferentialEquations.ContinuousCallback(OpticalBlochEquations.condition_new, OpticalBlochEquations.stochastic_collapse_new!, save_positions = (false, false))
cbs_diffusion = DifferentialEquations.CallbackSet(cb1_diffusion)

kwargs = (alg = DifferentialEquations.DP5(), reltol = 1e-4, saveat = 1000, maxiters = 200000000, callback = cbs_diffusion)
prob_SFcooling_diffusion = DifferentialEquations.ODEProblem(OpticalBlochEquations.ψ_fast_ballistic!, p_SFcooling_diffusion.u0, sim_type.(t_span), p_SFcooling_diffusion; kwargs...)

# set the total saturation
prob_SFcooling.p.sim_params.total_sat = sum(sats)
prob_SFcooling_diffusion.p.sim_params.total_sat = sum(sats)
