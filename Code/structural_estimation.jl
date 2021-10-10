using Distributed

#Pkg.add("PyPlot")
#Pkg.add("Distributions")
#Pkg.add("QuantEcon")
#Pkg.add("LaTeXStrings")
#Pkg.add("Optim")
#Pkg.add("ReadStat")
#Pkg.add("LinearAlgebra")
#Pkg.add("Interpolations")
#Pkg.add("FileIO")
#Pkg.add("StatFiles")
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("NamedArrays")
#Pkg.add("SharedArrrays")
#Pkg.add("DelimitedFiles")
#Pkg.add("Random")

@everywhere using PyPlot
@everywhere using Distributions
@everywhere using QuantEcon
@everywhere using LaTeXStrings
@everywhere using Optim
@everywhere using Optim: optimize
@everywhere using ReadStat
@everywhere using Interpolations
@everywhere using FileIO, StatFiles, DataFrames
@everywhere using CSV
@everywhere using LinearAlgebra
@everywhere using NamedArrays
@everywhere using SharedArrays
@everywhere using DelimitedFiles
@everywhere using Random

# Define Bellman operator

@everywhere function objective(c::AbstractFloat,
                                s::AbstractFloat,
                                W::AbstractFloat,
                                w_func::Interpolations.Extrapolation,
                                epsilon::Vector{Float64},
                                other_inc::Vector{Float64},
                                rho::AbstractFloat,
                                beta::AbstractFloat,
                                beta_m::AbstractFloat,
                                r::AbstractFloat
                                )
  if W > c + s && c > 0 && s > 0
    objective = - c ^ (1 - rho) / (1 - rho) - beta * mean(w_func((W - c - s) ^ beta_m * epsilon .+ other_inc .+ (1 + r) * s))
  else
    objective = 1e+10
  end
  return objective
end

@everywhere function bellman_operator_se(w::SharedArray{Float64}, 
                           grid_W::SharedArray{Float64}, 
                           beta::AbstractFloat,
                           beta_m::AbstractFloat,
                           r::AbstractFloat,
                           rho::AbstractFloat,
                           shocks::Array{Float64, 2},
                           Tw::SharedArray{Float64},
                           C::SharedArray{Float64},
                           M::SharedArray{Float64},
                           S::SharedArray{Float64}
                           )

  epsilon = shocks[:,1]
  other_inc = shocks[:,2]

  w_func = extrapolate(interpolate((grid_W,), w, Gridded(Linear())), Flat())

  @sync @distributed for i = 1:length(grid_W)
    CS = optimize(x -> objective(x[1], x[2], grid_W[i], w_func, epsilon, other_inc, rho, beta, beta_m, r), [1e-10, 1e-10])
    C[i] = CS.minimizer[1]
    S[i] = CS.minimizer[2]
    Tw[i] = - CS.minimum
    M[i] = grid_W[i] - C[i] - S[i]
  end

  return Tw, C, M, S
end
        
beta          = 0.96
beta_m        = 0.435
r             = 0.05
rho           = 1.3
mu_epsilon    = 3.4
var_epsilon   = 0.6
mu_other_inc  = 6.6
var_other_inc = 0.77

B = 200 # Number of simulations
grid_size = 200
grid_W = SharedArray{Float64}(exp10.(range(log10(10), stop=log10(10000), length=grid_size)))
shock_size = 25
Random.seed!(10)
shocks = Array{Float64, 2}(hcat(exp.(mu_epsilon .+ var_epsilon ^ 0.5 * randn(shock_size)), exp.(mu_other_inc .+ var_other_inc ^ 0.5 * randn(shock_size))))

w    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
Tw   = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
C    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
M    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
S    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)

@time w_new, C, M, S = bellman_operator_se(w, grid_W, beta, beta_m, r, rho, shocks, Tw, C, M, S)
goopy

@everywhere function solve_model(grid_W::SharedArray{Float64}, 
                      beta::AbstractFloat,
                      beta_m::AbstractFloat,
                      r::AbstractFloat,
                      rho::AbstractFloat,
                      shocks::Array{Float64, 2},
                      tol::AbstractFloat=1e-10,
                      max_iter::Integer=1000
                      )
                      
  w    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
  Tw   = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
  C    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
  M    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)
  S    = SharedArray{Float64}(1.0 .* ones(grid_size) .+ 0.001)

  error = tol + 1
  i = 0
  
  while (error > tol) && (i < max_iter)
    w_new, C, M, S = bellman_operator_se(w,
                              grid_W,
                              beta,
                              beta_m,
                              r,
                              rho,
                              shocks,
                              Tw,
                              C,
                              M,
                              S)
    error = maximum(abs, w_new - w)
    w = SharedArray{Float64}(copy(w_new))
    i += 1
    print("iteration = $i, error = $error \n" )
  end

  g = [C, M, S]

  value_func = extrapolate(interpolate((grid_W,), w,   Gridded(Linear())), Flat())
  C_func     = extrapolate(interpolate((grid_W,), C,   Gridded(Linear())), Flat())
  S_func     = extrapolate(interpolate((grid_W,), S,   Gridded(Linear())), Flat())

  return w, g, i, value_func, C_func, S_func
end

@time w, g, i, value_func, C_func, S_func = solve_model(grid_W, beta, beta_m, r, rho, shocks)
goopy

# Generate simulated data

nHH = 1000;
T = 4;

W_init = rand(Uniform(), nHH) * 1000 .+ 1;

W_data = Array{Float64, 2}(undef, nHH, T+1);
C_data = Array{Float64, 2}(undef, nHH, T+1);
S_data = Array{Float64, 2}(undef, nHH, T+1);
M_data = Array{Float64, 2}(undef, nHH, T+1);
Y_data = Array{Float64, 2}(undef, nHH, T+1);

W_data[:,1] = W_init[:];
C_data[:,1] = C_func(W_init);
S_data[:,1] = S_func(W_init);
M_data[:,1] = max.(W_data[:,1] - C_data[:,1] - S_data[:,1], 0.0001);
Y_data[:,1] = zeros(nHH);

for t in 1:T
  Random.seed!(10+t)
  epsilon = exp.(mu_epsilon .+ var_epsilon ^ 0.5 * randn(nHH))
  other_inc = exp.(mu_other_inc .+ var_other_inc ^ 0.5 * randn(nHH))
  Yp = M_data[:,t] .^ beta_m .* epsilon
  Y_data[:,(t+1)] = Yp
  Wp = max.((W_data[:,t] - C_data[:,t] - S_data[:,t]), ones(nHH) .* 0.0001) .^ beta_m .* epsilon .+ other_inc .+ (1 + r) * S_data[:,t]
  W_data[:,(t+1)] = Wp
  C_data[:,(t+1)] = C_func(Wp)
  S_data[:,(t+1)] = S_func(Wp)
  M_data[:,(t+1)] = max.((Wp - C_data[:,(t+1)] - S_data[:,(t+1)]), ones(nHH) .* 0.0001)
end

# Define objective function for indirect inference ----------------------
@everywhere function object_ii(param::Vector{Float64},
                    W_data::Array{Float64, 2},
                    C_data::Array{Float64, 2},
                    S_data::Array{Float64, 2},
                    M_data::Array{Float64, 2},
                    weight::Array{Float64, 2},
                    B::Integer,
                    nHH::Integer,
                    value_func::Interpolations.Extrapolation,
                    C_func::Interpolations.Extrapolation,
                    S_func::Interpolations.Extrapolation,
                    obtain_weight_index = 0)
    # parameter values
    beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc = param
    
    # Random variables (productivity shock and other income)
    epsilon   = exp.(mu_epsilon   .+ var_epsilon   ^ 0.5 * randn((nHH, T, B)));
    epsilon2  = exp.(mu_epsilon   .+ var_epsilon   ^ 0.5 * randn((nHH, T, B)));
    other_inc = exp.(mu_other_inc .+ var_other_inc ^ 0.5 * randn((nHH, T, B)));
    # Simulate current consumption, input, and saving
    C_model    = similar(C_data)
    S_model    = similar(S_data)
    M_model    = similar(M_data)

    C_model = max.(C_func.(W_data), 0.0001)
    S_model = max.(S_func.(W_data), 0.0001)
    M_model = max.(W_data .- C_model .- S_model, 0.0001)

    # Simulate next period production, wealth, saving, consumption, and input
    Yp_model  = zeros(nHH, T)
    Wp_model  = zeros(nHH, T)
    Sp_model  = zeros(nHH, T)
    Cp_model  = zeros(nHH, T)
    Mp_model  = zeros(nHH, T)

    EYp = zeros(nHH, T)
    EWp = zeros(nHH, T)
    ESp = zeros(nHH, T)
    ECp = zeros(nHH, T)
    EMp = zeros(nHH, T)

    for s in 1:B
      # Next period sales, wealth, and capital
      Yp_model = M_model[:,1:T] .^ beta_m .* epsilon2[:,:,s]
      Wp_model = Yp_model .+ other_inc[:,:,s] .+ (1 + r) * S_model[:,1:T]
      Sp_model = max.(S_func.(Wp_model), 0.0001)
      Cp_model = max.(C_func.(Wp_model), 0.0001)
      Mp_model = max.(Wp_model .- Cp_model .- Sp_model, 0.0001)
      EYp = EYp .+ Yp_model
      EWp = EWp .+ Wp_model
      ESp = ESp .+ Sp_model
      ECp = ECp .+ Cp_model
      EMp = EMp .+ Mp_model
    end

    EYp = EYp / B
    EWp = EWp / B
    ESp = ESp / B
    ECp = ECp / B
    EMp = EMp / B
    
    # Calculate deviations
    eps_Cp      = C_data[:,2:(T+1)] .- Cp_model
    eps_Cp_var  = (C_data[:,2:(T+1)] .- mean(C_data[:,2:(T+1)])) .^ 2 .- (Cp_model .- mean(Cp_model)) .^ 2
    eps_CpWp    = (C_data[:,2:(T+1)] .- Cp_model) .* W_data[:,2:(T+1)]

    eps_Sp      = S_data[:,2:(T+1)] .- Sp_model
    eps_Sp_var  = (S_data[:,2:(T+1)] .- mean(S_data[:,2:(T+1)])) .^ 2 .- (Sp_model .- mean(Sp_model)) .^ 2
    eps_SpWp    = (S_data[:,2:(T+1)] .- Sp_model) .* W_data[:,2:(T+1)]

    eps_Mp      = M_data[:,2:(T+1)] .- Mp_model
    eps_Mp_var  = (M_data[:,2:(T+1)] .- mean(M_data[:,2:(T+1)])) .^ 2 .- (Mp_model .- mean(Mp_model)) .^ 2
    eps_MpWp    = (M_data[:,2:(T+1)] .- Mp_model) .* W_data[:,2:(T+1)]

    eps_Yp     =  Y_data[:,2:(T+1)] .- EYp
    eps_Yp_var = (Y_data[:,2:(T+1)] .- mean(Y_data[:,2:(T+1)])) .^ 2 .- (EYp .- mean(EYp)) .^ 2

    eps_Wp     =  W_data[:,2:(T+1)] .- EWp
    eps_Wp_var = (W_data[:,2:(T+1)] .- mean(W_data[:,2:(T+1)])) .^ 2 .- (EWp .- mean(EWp)) .^ 2

    if obtain_weight_index == 1
      gi_hat_gi_hatp = Array{Float64, 2}(zeros(13, 13))
      gn_bar = Vector{Float64}(zeros(13))
      gi_hat = Vector{Float64}(zeros(13))
      i = 1
      j = 1
      for i in 1:nHH
        for j in 1:T
          gi_hat = [eps_Cp[i,j], eps_Cp_var[i,j], eps_CpWp[i,j], 
                    eps_Sp[i,j], eps_Sp_var[i,j], eps_SpWp[i,j], 
                    eps_Mp[i,j], eps_Mp_var[i,j], eps_MpWp[i,j], 
                    eps_Yp[i,j], eps_Yp_var[i,j], 
                    eps_Wp[i,j], eps_Wp_var[i,j]]
          gn_bar = gn_bar .+ gi_hat
          gi_hat_gi_hatp = gi_hat_gi_hatp .+ gi_hat * gi_hat'
        end
      end
      gn_bar = gn_bar / (nHH * T)
      gn_bar_gn_barp = gn_bar * gn_bar'
      weight_output = gi_hat_gi_hatp / (nHH * T) .- gn_bar_gn_barp
      return inv(weight_output)
    else
      # Calculate moments

      mom_Cp      = mean(eps_Cp)
      mom_Cp_var  = mean(eps_Cp_var)
      mom_CpWp    = mean(eps_CpWp)

      mom_Sp      = mean(eps_Sp)
      mom_Sp_var  = mean(eps_Sp_var)
      mom_SpWp    = mean(eps_SpWp)

      mom_Mp      = mean(eps_Mp)
      mom_Mp_var  = mean(eps_Mp_var)
      mom_MpWp    = mean(eps_MpWp)

      mom_Yp     = mean(eps_Yp)
      mom_Yp_var = mean(eps_Yp_var)

      mom_Wp     = mean(eps_Wp)
      mom_Wp_var = mean(eps_Wp_var)

      mom = [mom_Cp, mom_Cp_var, mom_CpWp, 
             mom_Sp, mom_Sp_var, mom_SpWp, 
             mom_Mp, mom_Mp_var, mom_MpWp, 
             mom_Yp, mom_Yp_var, 
             mom_Wp, mom_Wp_var]

      return sqrt(dot(mom, weight, mom))
    end
end

param = [beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc]
weight = Matrix{Float64}(I, 13, 13)

@time object_ii(param, W_data, C_data, S_data, M_data, weight, B, nHH, value_func, C_func, S_func)

@everywhere function whole_loop(parameter::Vector{Float64},
                     W_data::Array{Float64, 2},
                     C_data::Array{Float64, 2},
                     S_data::Array{Float64, 2},
                     M_data::Array{Float64, 2},
                     weight::Array{Float64, 2},
                     grid_W::SharedArray{Float64},
                     nHH::Integer,
                     B::Integer,
                     param_lb::Vector{Float64},
                     param_ub::Vector{Float64})
    # Expand parameters
    beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc = parameter

    beta          = param_lb[1] + exp(beta         ) / (1 + exp(beta         )) * (param_ub[1] - param_lb[1])
    beta_m        = param_lb[2] + exp(beta_m       ) / (1 + exp(beta_m       )) * (param_ub[2] - param_lb[2])
    r             = param_lb[3] + exp(r            ) / (1 + exp(r            )) * (param_ub[3] - param_lb[3])
    rho           = param_lb[4] + exp(rho          ) / (1 + exp(rho          )) * (param_ub[4] - param_lb[4])
    mu_epsilon    = param_lb[5] + exp(mu_epsilon   ) / (1 + exp(mu_epsilon   )) * (param_ub[5] - param_lb[5])
    var_epsilon   = param_lb[6] + exp(var_epsilon  ) / (1 + exp(var_epsilon  )) * (param_ub[6] - param_lb[6])
    mu_other_inc  = param_lb[7] + exp(mu_other_inc ) / (1 + exp(mu_other_inc )) * (param_ub[7] - param_lb[7])
    var_other_inc = param_lb[8] + exp(var_other_inc) / (1 + exp(var_other_inc)) * (param_ub[8] - param_lb[8])

    parameter = [beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc]
    print("$parameter \n")

    # Generate random shocks
    Random.seed!(4)
    shocks = Array{Float64, 2}(hcat(exp.(mu_epsilon .+ var_epsilon ^ 0.5 * randn(shock_size)), exp.(mu_other_inc .+ var_other_inc ^ 0.5 * randn(shock_size))))

    # Solve dynamic programming 
    value, g, i, value_func, C_func, S_func = solve_model(grid_W, beta, beta_m, r, rho, shocks)

    # Simulate data and obtain objective function
    objective_value = object_ii(parameter, W_data, C_data, S_data, M_data, weight, B, nHH, value_func, C_func, S_func)

    # Return objective value
    print("Objective value is $objective_value \n")
    return objective_value
end

param_lb  = [0.80 , 0.30 , 0.01 , 1.1 , 2.0 , 0.1 , 5.0 , 0.5]
param_ub  = [0.99 , 0.60 , 0.10 , 1.7 , 5.0 , 1.5 , 8.0 , 1.5]
parameter_true = [beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc]
parameter = [beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc] * 0.9

init_parameter = [log((parameter[1] - param_lb[1]) / (param_ub[1] - parameter[1])),
               log((parameter[2] - param_lb[2]) / (param_ub[2] - parameter[2])),
               log((parameter[3] - param_lb[3]) / (param_ub[3] - parameter[3])),
               log((parameter[4] - param_lb[4]) / (param_ub[4] - parameter[4])),
               log((parameter[5] - param_lb[5]) / (param_ub[5] - parameter[5])),
               log((parameter[6] - param_lb[6]) / (param_ub[6] - parameter[6])),
               log((parameter[7] - param_lb[7]) / (param_ub[7] - parameter[7])),
               log((parameter[8] - param_lb[8]) / (param_ub[8] - parameter[8]))] # Initial guess

@time results_first = optimize(x -> whole_loop(x, W_data, C_data, S_data, M_data, weight, grid_W, nHH, B, param_lb, param_ub), init_parameter, Optim.NelderMead(), Optim.Options(time_limit = 60.0 * 60.0 * 72, show_trace=true))
results_param = results_first.minimizer
#results_param = init_parameter

beta_est          = param_lb[1] + exp(results_param[1]) / (1 + exp(results_param[1])) * (param_ub[1] - param_lb[1])
beta_m_est        = param_lb[2] + exp(results_param[2]) / (1 + exp(results_param[2])) * (param_ub[2] - param_lb[2])
r_est             = param_lb[3] + exp(results_param[3]) / (1 + exp(results_param[3])) * (param_ub[3] - param_lb[3])
rho_est           = param_lb[4] + exp(results_param[4]) / (1 + exp(results_param[4])) * (param_ub[4] - param_lb[4])
mu_epsilon_est    = param_lb[5] + exp(results_param[5]) / (1 + exp(results_param[5])) * (param_ub[5] - param_lb[5])
var_epsilon_est   = param_lb[6] + exp(results_param[6]) / (1 + exp(results_param[6])) * (param_ub[6] - param_lb[6])
mu_other_inc_est  = param_lb[7] + exp(results_param[7]) / (1 + exp(results_param[7])) * (param_ub[7] - param_lb[7])
var_other_inc_est = param_lb[8] + exp(results_param[8]) / (1 + exp(results_param[8])) * (param_ub[8] - param_lb[8])

result_rescale_first = [beta_est, beta_m_est, r_est, rho_est, mu_epsilon_est, var_epsilon_est, mu_other_inc_est, var_other_inc_est]
#writedlm("results_coef_first.csv", result_rescale_first)

# --------------------------------- #

@everywhere function obtain_weight(parameter::Vector{Float64},
                     W_data::Array{Float64, 2},
                     C_data::Array{Float64, 2},
                     S_data::Array{Float64, 2},
                     M_data::Array{Float64, 2},
                     grid_W::SharedArray{Float64},
                     nHH::Integer,
                     B::Integer,
                     param_lb::Vector{Float64},
                     param_ub::Vector{Float64})
    # Expand parameters
    beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc = parameter

    beta          = param_lb[1] + exp(beta         ) / (1 + exp(beta         )) * (param_ub[1] - param_lb[1])
    beta_m        = param_lb[2] + exp(beta_m       ) / (1 + exp(beta_m       )) * (param_ub[2] - param_lb[2])
    r             = param_lb[3] + exp(r            ) / (1 + exp(r            )) * (param_ub[3] - param_lb[3])
    rho           = param_lb[4] + exp(rho          ) / (1 + exp(rho          )) * (param_ub[4] - param_lb[4])
    mu_epsilon    = param_lb[5] + exp(mu_epsilon   ) / (1 + exp(mu_epsilon   )) * (param_ub[5] - param_lb[5])
    var_epsilon   = param_lb[6] + exp(var_epsilon  ) / (1 + exp(var_epsilon  )) * (param_ub[6] - param_lb[6])
    mu_other_inc  = param_lb[7] + exp(mu_other_inc ) / (1 + exp(mu_other_inc )) * (param_ub[7] - param_lb[7])
    var_other_inc = param_lb[8] + exp(var_other_inc) / (1 + exp(var_other_inc)) * (param_ub[8] - param_lb[8])

    parameter = [beta, beta_m, r, rho, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc]
    print("$parameter \n")

    # Generate random shocks
    Random.seed!(4)
    shocks = Array{Float64, 2}(hcat(exp.(mu_epsilon .+ var_epsilon ^ 0.5 * randn(shock_size)), exp.(mu_other_inc .+ var_other_inc ^ 0.5 * randn(shock_size))))

    # Solve dynamic programming 
    value, g, i, value_func, C_func, S_func = solve_model(grid_W, beta, beta_m, r, rho, shocks)

    # Simulate data and obtain objective function
    weight_temp = Matrix{Float64}(I, 13, 13)
    weight_output = object_ii(parameter, W_data, C_data, S_data, M_data, weight, B, nHH, value_func, C_func, S_func, 1)

    # Return objective value
    return weight_output
end

new_weight = obtain_weight(results_param, W_data, C_data, S_data, M_data, grid_W, nHH, B, param_lb, param_ub)

#parameter_test = [log((parameter_true[1] - param_lb[1]) / (param_ub[1] - parameter_true[1])),
#               log((parameter_true[2] - param_lb[2]) / (param_ub[2] - parameter_true[2])),
#               log((parameter_true[3] - param_lb[3]) / (param_ub[3] - parameter_true[3])),
#               log((parameter_true[4] - param_lb[4]) / (param_ub[4] - parameter_true[4])),
#               log((parameter_true[5] - param_lb[5]) / (param_ub[5] - parameter_true[5])),
#               log((parameter_true[6] - param_lb[6]) / (param_ub[6] - parameter_true[6])),
#               log((parameter_true[7] - param_lb[7]) / (param_ub[7] - parameter_true[7])),
#               log((parameter_true[8] - param_lb[8]) / (param_ub[8] - parameter_true[8]))] # Initial guess
#new_weight = obtain_weight(parameter_test, W_data, C_data, S_data, M_data, grid_W, nHH, B, param_lb, param_ub)

@time results = Optim.optimize(x -> whole_loop(x, W_data, C_data, S_data, M_data, new_weight, grid_W, nHH, B, param_lb, param_ub), init_parameter, Optim.NelderMead(), Optim.Options(time_limit = 60.0 * 60.0 * 72, show_trace=true))
results_param = results.minimizer

beta_est          = param_lb[1] + exp(results_param[1]) / (1 + exp(results_param[1])) * (param_ub[1] - param_lb[1])
beta_m_est        = param_lb[2] + exp(results_param[2]) / (1 + exp(results_param[2])) * (param_ub[2] - param_lb[2])
r_est             = param_lb[3] + exp(results_param[3]) / (1 + exp(results_param[3])) * (param_ub[3] - param_lb[3])
rho_est           = param_lb[4] + exp(results_param[4]) / (1 + exp(results_param[4])) * (param_ub[4] - param_lb[4])
mu_epsilon_est    = param_lb[5] + exp(results_param[5]) / (1 + exp(results_param[5])) * (param_ub[5] - param_lb[5])
var_epsilon_est   = param_lb[6] + exp(results_param[6]) / (1 + exp(results_param[6])) * (param_ub[6] - param_lb[6])
mu_other_inc_est  = param_lb[7] + exp(results_param[7]) / (1 + exp(results_param[7])) * (param_ub[7] - param_lb[7])
var_other_inc_est = param_lb[8] + exp(results_param[8]) / (1 + exp(results_param[8])) * (param_ub[8] - param_lb[8])

result_rescale = [beta_est, beta_m_est, r_est, rho_est, mu_epsilon_est, var_epsilon_est, mu_other_inc_est, var_other_inc_est]

output_matrix = [parameter_true parameter result_rescale_first result_rescale]
writedlm("result_coef.csv", output_matrix)

goopy


# --------------------------------- #

# Bootstrap to estimate standard errors

B = 50
bootstrap_coef = []
for b in 1:B
  Random.seed!(b)
  bootstrap_sample = sample(1:nHH, nHH)
  bootstrap_data = data_ghana_control[bootstrap_sample, :]

  bootstrap_results_first = optimize(x -> whole_loop(x, W_data, C_data, S_data, M_data, weight, grid_W, nHH, B, param_lb, param_ub), init_parameter, Optim.NelderMead(), Optim.Options(time_limit = 60.0 * 60.0 * 72, show_trace=true))
  bootstrap_results_param = bootstrap_results_first.minimizer
  
  bootstrap_beta_est          = param_lb[1] + exp(bootstrap_results_param[1]) / (1 + exp(bootstrap_results_param[1])) * (param_ub[1] - param_lb[1])
  bootstrap_beta_m_est        = param_lb[2] + exp(bootstrap_results_param[2]) / (1 + exp(bootstrap_results_param[2])) * (param_ub[2] - param_lb[2])
  bootstrap_r_est             = param_lb[3] + exp(bootstrap_results_param[3]) / (1 + exp(bootstrap_results_param[3])) * (param_ub[3] - param_lb[3])
  bootstrap_rho_est           = param_lb[4] + exp(bootstrap_results_param[4]) / (1 + exp(bootstrap_results_param[4])) * (param_ub[4] - param_lb[4])
  bootstrap_mu_epsilon_est    = param_lb[5] + exp(bootstrap_results_param[5]) / (1 + exp(bootstrap_results_param[5])) * (param_ub[5] - param_lb[5])
  bootstrap_var_epsilon_est   = param_lb[6] + exp(bootstrap_results_param[6]) / (1 + exp(bootstrap_results_param[6])) * (param_ub[6] - param_lb[6])
  bootstrap_mu_other_inc_est  = param_lb[7] + exp(bootstrap_results_param[7]) / (1 + exp(bootstrap_results_param[7])) * (param_ub[7] - param_lb[7])
  bootstrap_var_other_inc_est = param_lb[8] + exp(bootstrap_results_param[8]) / (1 + exp(bootstrap_results_param[8])) * (param_ub[8] - param_lb[8])

  bootstrap_result_rescale_first = [bootstrap_beta_est, bootstrap_beta_m_est, bootstrap_r_est, bootstrap_rho_est, bootstrap_mu_epsilon_est, bootstrap_var_epsilon_est, bootstrap_mu_other_inc_est, bootstrap_var_other_inc_est]

  new_weight = obtain_weight(bootstrap_results_param, W_data, C_data, S_data, M_data, grid_W, nHH, B, param_lb, param_ub)

  bootstrap_results = Optim.optimize(x -> whole_loop(x, W_data, C_data, S_data, M_data, new_weight, grid_W, nHH, B, param_lb, param_ub), init_parameter, Optim.NelderMead(), Optim.Options(time_limit = 60.0 * 60.0 * 72, show_trace=true))
  bootstrap_results_param = bootstrap_esults.minimizer
  
  bootstrap_beta_est          = param_lb[1] + exp(bootstrap_results_param[1]) / (1 + exp(bootstrap_results_param[1])) * (param_ub[1] - param_lb[1])
  bootstrap_beta_m_est        = param_lb[2] + exp(bootstrap_results_param[2]) / (1 + exp(bootstrap_results_param[2])) * (param_ub[2] - param_lb[2])
  bootstrap_r_est             = param_lb[3] + exp(bootstrap_results_param[3]) / (1 + exp(bootstrap_results_param[3])) * (param_ub[3] - param_lb[3])
  bootstrap_rho_est           = param_lb[4] + exp(bootstrap_results_param[4]) / (1 + exp(bootstrap_results_param[4])) * (param_ub[4] - param_lb[4])
  bootstrap_mu_epsilon_est    = param_lb[5] + exp(bootstrap_results_param[5]) / (1 + exp(bootstrap_results_param[5])) * (param_ub[5] - param_lb[5])
  bootstrap_var_epsilon_est   = param_lb[6] + exp(bootstrap_results_param[6]) / (1 + exp(bootstrap_results_param[6])) * (param_ub[6] - param_lb[6])
  bootstrap_mu_other_inc_est  = param_lb[7] + exp(bootstrap_results_param[7]) / (1 + exp(bootstrap_results_param[7])) * (param_ub[7] - param_lb[7])
  bootstrap_var_other_inc_est = param_lb[8] + exp(bootstrap_results_param[8]) / (1 + exp(bootstrap_results_param[8])) * (param_ub[8] - param_lb[8])

  bootstrap_result_rescale = [bootstrap_beta_est, bootstrap_beta_m_est, bootstrap_r_est, bootstrap_rho_est, bootstrap_mu_epsilon_est, bootstrap_var_epsilon_est, bootstrap_mu_other_inc_est, bootstrap_var_other_inc_est]

  push!(bootstrap_coef, bootstrap_result_rescale)
  print("Bootstrap estimation result (b = $b) is $bootstrap_result_rescale \n")
end
goopy

bootstrap_se = zeros(length(results_param))
for i in 1:length(results_param)
  bootstrap_centered = 0
  for b in 1:B
    bootstrap_centered += (bootstrap_coef[b][i] - result_rescale[i]) ^ 2
    bootstrap_se[i] = sqrt(bootstrap_centered) / B
  end
end

writedlm("result_bootstrap_se.csv", bootstrap_se)

results_coef_se = round.(hcat(result_rescale, bootstrap_se), 3)
results_coef_se = ["Coefficients" "SE"; results_coef_se]
results_coef_se = hcat(["Parameters", "\$\\beta\$", "\$\\beta_m\$", "\$\\beta_k\$", "\$\\rho\$", "\$I\$", "\$\\mu_{\\epsilon}\$", "\$\\sigma_{\\epsilon}\$", "\$\\mu_Z\$", "\$\\sigma_Z\$"], results_coef_se)
writedlm("results_coef_se_0509.csv", results_coef_se, " , ")

# Show moment deviations 

@everywhere function simulation_mom(param::Vector{Float64},
                    data::DataFrame,
                    S::Integer,
                    nHH::Integer,
                    value_func::Interpolations.Extrapolation,
                    C0_func::Interpolations.Extrapolation,
                    C1_func::Interpolations.Extrapolation)
    # parameter values
    beta, beta_m, beta_k, rho, I, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc = param
    
    # Random variables (productivity shock and other income) for expectation calculations
    d = 4 # Number of shocks
    Sval, Sw = qnwlogn([d, d],[mu_epsilon, mu_other_inc],[var_epsilon 0.0; 0.0 var_other_inc])
    ds = length(Sw)
    # Random variables (productivity shock and other income)
    epsilon   = exp.(mu_epsilon   + var_epsilon   ^ 0.5 * randn((nHH,S)));
    epsilon2  = exp.(mu_epsilon   + var_epsilon   ^ 0.5 * randn((nHH,S)));
    other_inc = exp.(mu_other_inc + var_other_inc ^ 0.5 * randn((nHH,S)));
    # Simulate current consumption, intermediate goods, and investestment
    C_model    = similar(data[:K])
    M_model    = similar(data[:K])
    Inv_model  = similar(data[:K])
    EC_model   = zeros(nHH)
    EM_model   = zeros(nHH)
    EInv_model = zeros(nHH)

    for j in 1:ds
      for i in 1:nHH
        if data[:W][i] <= I
          C_model[i] = C0_func[data[:K][i], data[:W][i]]
          Inv_model[i] = 0
        else
          C0_i = maximum([C0_func[data[:K][i], data[:W][i]], 0.0001])
          C1_i = maximum([C1_func[data[:K][i], data[:W][i]], 0.0001])
          Inv_model[i] = ((C1_i ^ (1 - rho) / (1 - rho) + beta * value_func[data[:K][i] + I, (data[:K][i] + I) ^ beta_k * maximum([data[:W][i] - I - C1_i, 0.01]) ^ beta_m * Sval[j,1] + Sval[j,2]]) >= 
                          (C0_i ^ (1 - rho) / (1 - rho) + beta * value_func[data[:K][i],      data[:K][i]      ^ beta_k * maximum([data[:W][i] -     C0_i, 0.01]) ^ beta_m * Sval[j,1] + Sval[j,2]]))
          C_model[i] = Inv_model[i] * C1_i + (1 - Inv_model[i]) * C0_i
        end
        M_model[i] = data[:W][i] - C_model[i] - Inv_model[i] * I
      end
      # Average for each household
      EC_model = EC_model + C_model * Sw[j]
      EM_model = EM_model + M_model * Sw[j]
      EInv_model = EInv_model + Inv_model * Sw[j]
    end

    # Simulate next period sales, wealth, and capital
    C_model   = similar(data[:K])
    M_model   = similar(data[:K])
    Inv_model = similar(data[:K])
    Yp_model  = zeros(nHH, S)
    Wp_model  = zeros(nHH, S)
    Kp_model  = zeros(nHH, S)

    for s in 1:S
      for i in 1:nHH
        if data[:W][i] <= I
          C_model[i] = C0_func[data[:K][i], data[:W][i]]
          Inv_model[i] = 0
        else
          C0_i = maximum([C0_func[data[:K][i], data[:W][i]], 0.0001])
          C1_i = maximum([C1_func[data[:K][i], data[:W][i]], 0.0001])
          Inv_model[i] = ((C1_i ^ (1 - rho) / (1 - rho) + beta * value_func[data[:K][i] + I, (data[:K][i] + I) ^ beta_k * maximum([data[:W][i] - I - C1_i, 0.01]) ^ beta_m * epsilon[i,s] + other_inc[i,s]]) >= (C0_i ^ (1 - rho) / (1 - rho) + beta * value_func[data[:K][i], data[:K][i] ^ beta_k * maximum([data[:W][i] - C0_i, 0.01]) ^ beta_m * epsilon[i,s] + other_inc[i,s]]))
          C_model[i] = Inv_model[i] * C1_i + (1 - Inv_model[i]) * C0_i
        end
        M_model[i] = data[:W][i] - C_model[i] - Inv_model[i] * I
        # Next period sales, wealth, and capital
        Yp_model[i,s] = (data[:K][i] + Inv_model[i] * I) ^ beta_k * M_model[i] ^ beta_m * epsilon2[i,s]
        Wp_model[i,s] = Yp_model[i,s] + other_inc[i,s]
        Kp_model[i,s] = data[:K][i] + Inv_model[i] * I
      end
    end

    EYp = mean(Yp_model, 2)
    EWp = mean(Wp_model, 2)
    EKp = mean(Kp_model, 2)

    sim_mom = [mean(EC_model), mean(EM_model), mean(EInv_model), mean(EYp), mean(EWp), mean(EKp)]
    return sim_mom
end

@everywhere function mom_dev(parameter::Vector{Float64},
                     data::DataFrame,
                     grid_K::Vector{Float64},
                     grid_W::Vector{Float64},
                     nHH::Integer,
                     S::Integer;
                     B::Integer = 100)
    # parameter values
    beta, beta_m, beta_k, rho, I, mu_epsilon, var_epsilon, mu_other_inc, var_other_inc = parameter

    # Simulated moments (mean and standard deviations)
    # Generate random shocks
    sim_mom = []
    for b in 1:B
      Random.seed!(b)
      shocks = hcat(exp.(mu_epsilon + var_epsilon ^ 0.5 * randn(shock_size)), exp.(mu_other_inc + var_other_inc ^ 0.5 * randn(shock_size)))

      # Solve dynamic programming 
      value, g, invest, Inv_func, i, value_func, C_func, C0_func, C1_func = solve_model(grid_K, grid_W, beta, beta_m, beta_k, rho, shocks, I)

      # Simulate data and obtain objective function
      simulation_mom_b = simulation_mom(parameter, data, S, nHH, value_func, C0_func, C1_func)

      push!(sim_mom, simulation_mom_b)
    end
    
    sim_mom_mean = zeros(6)
    sim_mom_std  = zeros(6)
    for i in 1:6
      for b in 1:B
        sim_mom_mean[i] += sim_mom[b][i]
      end
      sim_mom_mean[i] = sim_mom_mean[i] / B
      for b in 1:B
        sim_mom_std[i]  += (sim_mom[b][i] - sim_mom_mean[i]) ^ 2
      end
      sim_mom_std[i]  = sim_mom_std[i] / B
    end
    
    sim_mom_mean_std = hcat(sim_mom_mean, sim_mom_std)

    # Data moments
    data_mom = [mean(data[:C]), mean(data[:Mp]), mean(data[:D]), mean(data[:Yp]), mean(data[data[:wave] .< 5, :][:Wp]), mean(data[:K] + data[:I])]

    # Tables of moments
    mom_dev_output = round.(hcat(data_mom, sim_mom_mean_std), 3)
    
    mom_dev_output = ["Data Moments" "Simulated Moments (mean)" "(Std)"; mom_dev_output]
    mom_dev_output = hcat(["Variables", "Consumption", "Intermediate inputs$", "Investment decision", "Next-period sales", "Next-period wealth", "Next-period capital"], mom_dev_output)

    return mom_dev_output
end

mom_dev_table = mom_dev(result_rescale, data_ghana_control, grid_K, grid_W, nHH, S)
writedlm("mom_dev.csv", mom_dev_table, " , ")


