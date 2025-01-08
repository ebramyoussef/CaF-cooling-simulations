### SIMULATION PARAMETERS: 3-FREQUENCY 1+2 BLUE MOT  ###
import QuantumStates
import UnitsToValue: c, ħ
include("define_CaF_molecular_structure.jl")
# DEFINE STATES #
energy_offset = (2π / Γ) * QuantumStates.energy(states[13])
energies = QuantumStates.energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = +1
δ1 = +0.00

Δ1 = 1e6 * (detuning + δ1)

f1 = QuantumStates.energy(states[end]) - QuantumStates.energy(states[1]) + Δ1

freqs = [f1] .* (2π / Γ)

# DEFINE SATURATION INTENSITIES #
beam_radius = 5e-3
Isat = π * h * c * Γ / (3*λ^3)
P = 0.55 * 13.1e-3 # 13.1 mW/1 V at 1.0 V, factor of 0.55 to match scattering rates
I = 2*P / (π * beam_radius^2)

total_sat = I / Isat
s1 = total_sat

sats = [s1]

# DEFINE POLARIZATIONS #
pols = [OpticalBlochEquations.σ⁻]

# DEFINE FUNCTION TO UPDATE PARAMETERS DURING SIMULATION #
@everywhere function update_p_SFcooling!(p, r, t)
    s = p.sim_params.total_sat
    s_factor = p.sim_params.s_factor_end
    p.sats[1] = s * s_factor
    return nothing
end

@everywhere function update_p_SFcooling_diffusion!(p, r, t)
    s = p.sim_params.total_sat
    s_factor = p.sim_params.s_factor_end
    p.sats[1] = s * s_factor
    return nothing
end