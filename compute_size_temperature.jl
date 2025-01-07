@everywhere function scan_nothing(prob, scan_value)
    return nothing
end

function compute_trajectories_with_diffusion(
        prob, prob_func, prob_diffusion, prob_func_diffusion, 
        n_trajectories1, n_trajectories2, n_trajectories_diffusion, 
        n_times, diffusion_t_end, diffusion_τ_total, scan_func=scan_nothing, scan_values=[0]
    )
    """
    1) Compute 
    2) 
    3)
    4)
    5)
    """

    (diffusions, diffusion_errors, diffusions_over_time) = distributed_compute_diffusion(
        prob_diffusion, prob_func_diffusion, n_trajectories_diffusion, diffusion_t_end, diffusion_τ_total, n_times, scan_func_with_initial_conditions!(scan_func), scan_values_with_σ_and_T
        )

    # 3)
    prob.p.add_spontaneous_decay_kick = false

    scan_values_with_diffusion = zip(scan_values, diffusions)
    all_sols_with_diffusion = distributed_solve(n_trajectories2, prob, prob_func, scan_func_with_diffusion!(scan_func), scan_values_with_diffusion)
    
    if length(scan_values) == 1
        return (all_sols_no_diffusion[1], all_sols_with_diffusion[1], diffusions[1], diffusion_errors[1], diffusions_over_time[1])
    else
        return (all_sols_no_diffusion, all_sols_with_diffusion, diffusions, diffusion_errors, diffusions_over_time)
    end
end

@everywhere function scan_func_with_initial_conditions!(scan_func)
    (prob, scan_value) -> begin
        scan_func(prob, scan_value[1])
        σx, σy, σz, Tx, Ty, Tz = scan_value[2]
        prob.p.sim_params.x_dist = Normal(0, σx)
        prob.p.sim_params.y_dist = Normal(0, σy)
        prob.p.sim_params.z_dist = Normal(0, σz)
        prob.p.sim_params.vx_dist = Normal(0, Tx)
        prob.p.sim_params.vy_dist = Normal(0, Ty)
        prob.p.sim_params.vz_dist = Normal(0, Tz)
        return nothing
    end
end

@everywhere function scan_func_with_diffusion!(scan_func)
    (prob, scan_value) -> begin
        scan_func(prob, scan_value[1])
        diffusion = scan_value[2]
        prob.p.diffusion_constant[1] = diffusion
        prob.p.diffusion_constant[2] = diffusion
        prob.p.diffusion_constant[3] = diffusion
        return nothing
    end
end

    
    