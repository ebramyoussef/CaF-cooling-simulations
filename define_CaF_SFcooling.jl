### SIMULATION PROBLEMS DEFINITION: 1+2 BLUE MOT ###
import OpticalBlochEquations
import QuantumStates
import DifferentialEquations
import Distributed
include("define_sim_params.jl")
include("CaF_SFcooling_params.jl")
# PROBLEM TO CALCULATE TRAJECTORIES #

t_start = 0.0
t_end   = 1e-3
t_span  = (t_start, t_end) ./ (1/Γ)

p_SFcooling = OpticalBlochEquations.initialize_prob(sim_type, energies, freqs, sats, pols, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p_SFcooling!, add_terms_dψ!)

cb1 = DifferentialEquations.ContinuousCallback(OpticalBlochEquations.condition_new, OpticalBlochEquations.stochastic_collapse_new!, save_positions=(false,false))
cb2 = DifferentialEquations.DiscreteCallback(terminate_condition, DifferentialEquations.terminate!)
cbs = DifferentialEquations.CallbackSet(cb1,cb2)

kwargs = (alg=DifferentialEquations.DP5(), reltol=1e-4, saveat=1000, maxiters=200000000, callback=cbs)
prob_SFcooling = DifferentialEquations.ODEProblem(ψ_fast!, p_SFcooling.u0, sim_type.(t_span), p_SFcooling; kwargs...)

# PROBLEM TO COMPUTE DIFFUSION #
p_SFcooling_diffusion = OpticalBlochEquations.initialize_prob(sim_type, energies, freqs, sats, pols, Inf, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p_SFcooling_diffusion!, add_terms_dψ!)

cb1_diffusion = DifferentialEquations.ContinuousCallback(OpticalBlochEquations.condition_new, OpticalBlochEquations.stochastic_collapse_new!, save_positions=(false,false))
cbs_diffusion = DifferentialEquations.CallbackSet(cb1_diffusion)

kwargs = (alg=DifferentialEquations.DP5(), reltol=1e-4, saveat=1000, maxiters=200000000, callback=cbs_diffusion)
prob_SFcooling_diffusion = DifferentialEquations.ODEProblem(ψ_fast_ballistic!, p_SFcooling_diffusion.u0, sim_type.(t_span), p_SFcooling_diffusion; kwargs...)

# set the total saturation
prob_SFcooling.p.sim_params.total_sat = sum(sats)
prob_SFcooling_diffusion.p.sim_params.total_sat = sum(sats)