using DataFrames
using Discretizers
using PyPlot
using StatsBase
using HypothesisTests

function separateData(data::DataFrame)
	# Separate data into true and simulated values
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
    j = zeros(length(acc) - 1)
    for i = 2:length(acc)
        j[i-1] = (acc[i] - acc[i - 1])/0.1
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
    dmn = 5.301
    T = 1.023
    bcmf = 3.621
    smx = 17.281
    
    amx = 3
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
    n = length(a)
    for i = 1:100:n-1
        for j = i + 1:i + 98
            if sign(a[j]) * sign(a[j - 1]) < 0
                count += 1
            end
        end
    end
    return count * 100 / n
end  

# Array of filenames
filenames = ["softmax.csv", "mixture.csv", "FF_softmax.csv", "../true_state/mixture.csv"]

# Initialize dictionaries to hold arrays of true and simulated quantities
sim = Dict()
real = Dict()
idm = Dict()

# Define horizon over which to plot RWSE
horizon = 5
m = 5

# Array of quantity names and units
names = ["Distance to Lead Vehicle", "Relative Speed", "Speed", "Acceleration", "Jerk"]
units = [" (m)", " (m/s)", " (m/s)", " (m/s^2)", " (m/s^3)"]

# Loop over files
for i = 1:length(filenames)
	println(filenames[i] * ": ")
	# Read in data from file
	data = readtable("./augment/" * filenames[i], header=false)
	total = length(data[:, 1])

	# Separate data into individual compnents
	sim[1], sim[2], sim[3], sim[4], real[1], real[2], real[3], real[4] = separateData(data)
	idm[1], idm[2], idm[3], idm[4] = getSimValues(real[1], real[2], real[3], real[4])

	# Get jerk values
	sim[5] = jerk(sim[4])
	real[5] = jerk(convert(Vector, real[4]))
	idm[5] = jerk(idm[4])

	println("Fraction of negative d values: $(length(sim[1][sim[1].< 0])/length(sim[1]))")

	if i == 1
		println("Fraction of real negative d values: $(length(real[1][real[1].< 0])/length(real[1]))")
		println("Fraction of IDM negative d values: $(length(idm[1][idm[1].< 0])/length(idm[1]))")
		println()
	end

	println("Fraction of negative speed values: $(length(sim[3][sim[3].< 0])/length(sim[3]))")

	if i == 1
		println("Fraction of real negative speed values: $(length(real[3][real[3].< 0])/length(real[3]))")
		println("Fraction of IDM negative speed values: $(length(idm[3][idm[3].< 0])/length(idm[3]))")
		println()
	end

	println("Jerk sign inversions per trajectory: $(countInversions(sim[5]))")

	if i == 1
		println("Real sign inversions per trajectory: $(countInversions(real[5]))")
		println("IDM sign inversions per trajectory: $(countInversions(idm[5]))")
		println()
	end

	# Loop over all quantities
	for j = 1:5
		println(names[j])
		println(KLDivergence(convert(Vector{Float64}, real[j]), sim[j], 100))

		# Generate plot
		figure(j)
		x = minimum(sim[j]):.1:maximum(sim[j])
		ef = ecdf(sim[j])
		if i == 1
			println("IDM: ")
			println(KLDivergence(convert(Vector{Float64}, real[j]), idm[j], 100))
			ef1 = ecdf(real[j])
			ef2 = ecdf(idm[j])
			plot(x, ef1(x), x, ef2(x), x, ef(x), linewidth=2.0);
		else
			plot(x, ef(x), linewidth=2.0);
		end

		# RWSE plot
		if j == 3
			figure(6)
			x = 1:horizon
			if i == 1
				plot(x, RWSE(idm[j], convert(Vector, real[j]), 1, horizon), "-o", linewidth=2.0);
			end
			plot(x, RWSE(sim[j], convert(Vector, real[j]), m, horizon), "-o", linewidth=2.0);
		end

		# Plot acceleration values throughout a trajectory
		if j == 4
			figure(7)
			t = 0.1:0.1:10 # time vector
			if i == 1
				plot(t, real[j][1:100], t, sim[j][101:200], linewidth=2.0);
			else
				plot(t, sim[j][101:200], linewidth=2.0);
			end
		end
		println()
	end
end

savenames = ["d", "r", "s", "a", "jerk"]
names_rwse = ["d_rwse", "r_rwse", "s_rwse", "a_rwse", "jerk_rwse"]
legend_labels = ["True", "IDM", "Softmax", "Mixture", "Feedforward Softmax", "Oracle"]
for i = 1:5
	figure(i)
	grid()
	plt[:rc]("text", usetex=true)
	plt[:rc]("font", family="serif")
	xlabel(names[i] * units[i])
	ylabel("Cumulative Probability")
	title("Empirical Distributions of $(names[i])")
	legend(legend_labels, loc="best")
	savefig("./" * savenames[i] * ".pdf")
end

figure(6)
grid()
plt[:rc]("text", usetex=true)
plt[:rc]("font", family="serif")
xlabel("Time Horizon (sec)", fontsize=16)
ylabel("RWSE (m/s)", fontsize=16)
# title("RWSE of Speed")
xticks(1:5)
legend(legend_labels[2:end], loc="best")
savefig("./" * names_rwse[3] * ".pdf")

figure(7)
grid()
plt[:rc]("text", usetex=true)
plt[:rc]("font", family="serif")
xlabel("Time (sec)")
ylabel("Acceleration (m/s^2)")
title("Acceleration in True and Simulated Trajectories")
legend(legend_labels, loc="best")
savefig("./accelerations.pdf")






