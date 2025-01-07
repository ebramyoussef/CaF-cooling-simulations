### SIMULATION PROBLEMS DEFINITION: 1+2 BLUE MOT ###

# PROBLEM TO CALCULATE TRAJECTORIES #
include("CaOH_SFcooling_params.jl")

t_start = 0.0
t_end   = 1e-3
t_span  = (t_start, t_end) ./ (1/Γ)

p_SFcooling = initialize_prob(sim_type, energies, freqs, sats, pols, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p_SFcooling!, add_terms_dψ!)

cb1 = ContinuousCallback(condition_new, stochastic_collapse_new!, save_positions=(false,false))
cb2 = DiscreteCallback(terminate_condition, terminate!)
cbs = CallbackSet(cb1,cb2)

kwargs = (alg=DP5(), reltol=1e-4, saveat=1000, maxiters=200000000, callback=cbs)
prob_SFcooling = ODEProblem(ψ_fast!, p_SFcooling.u0, sim_type.(t_span), p_SFcooling; kwargs...)

# PROBLEM TO COMPUTE DIFFUSION #
p_SFcooling_diffusion = initialize_prob(sim_type, energies, freqs, sats, pols, Inf, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p_SFcooling_diffusion!, add_terms_dψ!)

cb1_diffusion = ContinuousCallback(condition_new, stochastic_collapse_new!, save_positions=(false,false))
cbs_diffusion = CallbackSet(cb1_diffusion)

kwargs = (alg=DP5(), reltol=1e-4, saveat=1000, maxiters=200000000, callback=cbs_diffusion)
prob_SFcooling_diffusion = ODEProblem(ψ_fast_ballistic!, p_SFcooling_diffusion.u0, sim_type.(t_span), p_SFcooling_diffusion; kwargs...)

# set the total saturation
prob_SFcooling.p.sim_params.total_sat = sum(sats)
prob_SFcooling_diffusion.p.sim_params.total_sat = sum(sats)