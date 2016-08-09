using DataFrames
using Discretizers
using StatsBase

# Separate data into true and simulated values
function separateData(data::DataFrame)
	total_vals = length(data[1, :])
	sim_size = total_vals - 4
	sim_data = data[:, 1:sim_size]
	true_data = data[:, sim_size + 1:sim_size + 4]

	# Separate into individual state components
	sim_d = []
	for i = 1:4:sim_size
	    sim_d = vcat(sim_d, sim_data[:, i])
	end

	sim_r = []
	for i = 2:4:sim_size
	    sim_r = vcat(sim_r, sim_data[:, i])
	end

	sim_vel = []
	for i = 3:4:sim_size
	    sim_vel = vcat(sim_vel, sim_data[:, i])
	end

	sim_acc = []
	for i = 4:4:sim_size
	    sim_acc = vcat(sim_acc, sim_data[:, i])
	end

	# Get true values
	true_d = true_data[1:end-1, 1]
	true_r = true_data[1:end-1, 2]
	true_vel = true_data[:, 3]
	true_acc = true_data[:, 4]

	sim_d, sim_r, sim_vel, sim_acc, true_d, true_r, true_vel, true_acc
end

# Function to calculate jerk values -- approximated as change in acceleration over time step
function jerk(acc::Vector)
    # Find total number of trajectories
    n = convert(Int64, length(acc)/100)

    # Initialize vector to hold jerk values
    j = zeros(99*n)

    # Loop over trajectories
    for i = 1:n

        # Loop over entry in trajectory and find jerk
        for k = 2:100
            j[(i - 1) * 99 + k - 1] = (acc[(i - 1) * 100 + k] - acc[(i - 1) * 100 + k - 1])/0.1
        end
    end
    j
end 

# Calculate the root-weighted square error
function RWSE(sim::Vector, actual::Vector, m::Int64, horizon::Int64)
    err = zeros(horizon) # Vector to hold errors
    n = length(actual)
    for i = 0:100:n-1
        sum = zeros(horizon) # Sum of errors for simulated trajectories
        for j = 1:m
            for t = 1:horizon
                sum[t] += (actual[i + 10*t] - sim[i + 10*t + (m - 1)*n])^2
            end
        end
        sum /= m
        err += sum
    end
    err /= (length(actual)/100)
    sqrt(err) # Return RWSE
end    

# Intelligent driver model
function IDM(state::Array{Float64})
    # Define model parameters
    dmn = 5.249
    T = 0.918
    bcmf = 3.811
    smx = 17.837
    amx = 0.758

    d = state[1]
    r = state[2]
    s = state[3]
    
    # Find prediction and return
    d_des = dmn + T*s - s*r/2/sqrt(amx*bcmf)
    return amx*(1 - (s/smx)^4 - (d_des/d)^2)
end

# Simulate trajectories using IDM
function simulateIDM(initState::Vector{Float64})
    simState = zeros(99, 4)
    simState[1, :] = initState
    for i = 1:98
        acc = IDM(simState[i, 1:3])
        simState[i+1, 1] = simState[i, 1] + simState[i, 2]*0.1 - 0.5*0.01*simState[i, 4]
        simState[i+1, 2] = simState[i, 2] - simState[i, 4]*0.1
        simState[i+1, 3] = simState[i, 3] + simState[i, 4]*0.1
        simState[i+1, 4] = acc
    end
    return simState
end  

# Return vector of constant speed velocity values (i.e. the same velocity value repeated)
function simulateCS(initVel::Float64)
	return initVel * ones(99)
end

# Return vector of constant acceleration velocity values
function simulateCA(initState::Vector{Float64})
	return [initState[1] + 0.1*i*initState[2] for i = 0:98]
end

# Get KL Divergence for two distributions
function KLDivergence(real::Vector{Float64}, sim::Vector{Float64}, nbins::Int64)
    bin_bounds = linspace(minimum(real), maximum(real), nbins)
    lindisc = Discretizers.LinearDiscretizer(bin_bounds)
    simCounts = 0.5*ones(nbins) # Initialize counts to 0.5
    realCounts = 0.5*ones(nbins)
    
    for i = 1:length(sim)
        if i <= length(real)
            realCounts[Discretizers.encode(lindisc, real[i])] += 1
        end
        simCounts[Discretizers.encode(lindisc, sim[i])] += 1
    end
    
    realCounts /= sum(realCounts)
    simCounts /= sum(simCounts)
    
    kld = 0
    for i = 1:nbins
        kld += simCounts[i] * log(simCounts[i]/realCounts[i])
    end
    kld
end 

# Return simulated CS and CA values
function getCSCAValues(true_vel, true_acc)
    n = length(true_acc)
    CS_vel = zeros(n)
    CA_vel = zeros(n)
    for i = 1:100:n-1
        CS_vel[i:i+98] = simulateCS(true_vel[i])
        CA_vel[i:i+98] = simulateCA([true_vel[i], true_acc[i]])
    end
    CS_vel, CA_vel
end

# Return simulated IDM values
function getSimValues(true_d, true_r, true_vel, true_acc)
    n = length(true_acc)
    IDM_d = zeros(n)
    IDM_r = zeros(n)
    IDM_vel = zeros(n)
    IDM_acc = zeros(n)
    for i = 1:100:n-1
        IDMvals = simulateIDM([true_d[i], true_r[i], true_vel[i], true_acc[i]])
        IDM_d[i:i+98] = IDMvals[:, 1]
        IDM_r[i:i+98] = IDMvals[:, 2]
        IDM_vel[i:i+98] = IDMvals[:, 3]
        IDM_acc[i:i+98] = IDMvals[:, 4]
    end
    IDM_d, IDM_r, IDM_vel, IDM_acc
end

# Jerk sign inversions
function countInversions(a::Vector{Float64})
    count = 0
    j = jerk(a)
    n = length(j)
    for i = 1:99:n-1
        for k = i + 1:i + 98
            if sign(j[k]) * sign(j[k - 1]) < 0
                count += 1
            end
        end
    end
    return count * 99 / n
end  

# Return acceleration values for trajectories defined by indices
function retAccValues(trajValues::Vector{Float64}, idxs::Vector{Int64})
    start1 = (idxs[1] - 1)*100 + 1
    start2 = (idxs[2] - 1)*100 + 1
    start3 = (idxs[3] - 1)*100 + 1
    return trajValues[start1:start1+99], trajValues[start2:start2+99], trajValues[start3:start3+99]
end

# Get fraction of state values that are negative
function getFracNeg(states::Vector{Float64})
    return length(states[states .< 0])/length(states)
end
