### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 423cb435-ab09-426e-a29a-0b894fc767ba
begin
	using Downloads
	using CSV, DataFrames
	using MLBase, MLDataUtils

	using StatsBase, GLM
	using Flux
	using Flux: sigmoid, binarycrossentropy, logitbinarycrossentropy

	using Plots#, LaTeXStrings
	using ValueHistories
	using PlutoUI, ProgressLogging
end

# ╔═╡ 2671099c-667e-4c1b-9e5d-f41bd5752938
md"""
## Overview
In this lab, we'll revise the dataset of SDSS & Spitzer photometry with the goal of developing an improved classifier to identify high-${z}$ quasars.
We'll start by fitting logistic regression models using the framework of generalized linear models.  
Then, we'll train an equivalent neural network classifier.  
You'll get a chance to explore different strategies for training the model.  
Finally, you'll get a chance to explore more complex neural network architectures and see if you can build a significantly better neural network classifier.  

As before, we first need to read in the dataset.
"""

# ╔═╡ 783788a0-6f9a-4225-98e6-5030d9f21712
md"""
## Prepare the data
### Read data from file
"""

# ╔═╡ 1c792c1d-f1e8-4e5f-8a76-5c7ca5fb8587
md"""
### Create subsets of data for training & testing
We want to divide the dataset into two distinct subsets:  `df_cv` (a DataFrame of observations to be used with a cross-validation procedure) and `df_test` (a DataFrame of observations to be used to testing our final model), so that we can use one for model building and the second for testing our final model.  
Just in case there's any pattern in the order of the data, we'll shuffle the order of the observations _prior_ to assigning the observations to the two subsets.
The starter code places 66% of the observations into `df_cv` and the rest into `df_test`.  

Since high-${z}$ quasars are relatively rare, we'll make a point of using [stratified sampling](https://en.wikipedia.org/wiki/Stratified_sampling), so that both `df_cv` and `df_test` have (nearly) the same fraction of high-${z}$ quasars.

Similarly, since high-${z}$ quasars are relatively rare in our dataset, one might try undersampling the other objects, so that our dataset is nearly balanced.  
(Once you've worked through the first section of the notebook, you might want to come back and uncheck the box and see how the results change if you use an unbalanced training set.)
"""

# ╔═╡ 26635f45-1e34-4025-8151-2185d8d84e06
md"""
Undersample non-high-${z}$ quasars to make for balanced datasets?
$(@bind make_balanced CheckBox(default=true))
"""

# ╔═╡ ffe4cdc7-4863-4f0e-b790-4b86afcc56b8
md"""
### Constructing subset for K-fold cross-validation
Eventually, you'll be experimenting with multiple neural network architectures.  
We want to save the data in `df_test` for testing after we've finalized all our model choices.
In the mean time, we'll want to have subsets of the observations in `df_cv` for training and validating as part of a **k-fold cross-validation** process.

Below, you can choose how many "folds" to use and which fold will be used for the validation set.  That choice implicitly determine which n-1 folds will be used for training our logistic regression and neural network models.
"""

# ╔═╡ 7e0341a3-e52e-4a19-b9ac-969ebdd2161f
md"""
For convenience sake, we'll define several dataframes containing data based on your chosen fold.  
"""

# ╔═╡ 4aff0575-014d-4757-8e69-f49ad1f2e82e
md"""
## Logistic Regression Model
In this section, we'll use the general linear regression framework to efficiently fit a logistic regression model.
If you haven't worked through the [logistic regression application lab](), it's suggested you do that first.

We specify a formula specifying that we want to predict the data in column "label" using input data in all the other columns (and an offset term).
"""

# ╔═╡ 65472ed7-4c0a-4695-8dc3-94575a0241dd
md"And we fit a logistic regresion model to the selected training dataset."

# ╔═╡ b419a260-8ec8-46f9-bc5b-0fdd09717867
md"""
Below are the confusion matrices for the selected training and validation sets and the test dataset.  The bottom right is the fraction of objects that are both labeled and predicted to be high-${z}$ quasars.  The upper left is the fraction of objects both labeled and predicted to be other objects.  
"""

# ╔═╡ 6636da1a-64c0-4c12-b102-82a5d49df114
md"""
**Question:** How do these confusion matrices compare to those you obtained as part of the logistic regression lab?  

!!! hint "Hint"
    The false discovery rates and false omission rates were quite different in the previous lab.  Which of the extra steps we took this time is responsible for making them more similar?
"""

# ╔═╡ 7022d21e-10f3-46ac-a5fa-b4b6f7ab7881
md"""
### Compare results across folds
Since generalized linear regression is computationally efficient, we can go ahead and fit logistic regression models (one for each of the fold of data).  

Below we show histograms of the false discovery rates and false omission rates for each fold.  
The next cell shows the coefficients fit the training set for each fold.  
"""

# ╔═╡ f1e1d4ab-d1bf-4684-b3f6-3233b689ba71
md"We can compute the mean and standard deviation of each coefficient across the training data set for each different folds."

# ╔═╡ 54f52161-2337-40fc-a743-58cc0492726f
md"""
**Question:** How do the mean and standard deviation of coefficients from the cross validation procedure compare to the standard errors returned by the `glm` function above?  What factors could contribute to the differences?
"""

# ╔═╡ 84c733a4-5f23-42d6-bea5-c7cc89674d41
md"In the subsequent sections on neural networks, we'll use a loss function that is equal to negative log likelihood divided by the number of observations.  Let's compute that loss function for results of the standard logistic regression model on the training dataset for each fold, so we'll be able to make comparisons later."

# ╔═╡ bd6b7f5f-7f64-41ce-9ce8-639d6435800a
md"""
## Simple Neural Network for Logistic Regression
Logistic regression can be represented as a very simple neural network.  
Doing so will allow us to try out some of the tools and programming patterns used for training neural networks, while still getting (reasonably) prompt feedback.

To implement logistic regression as a neural network, we'll connect all six of the input parameters (six nodes in the input layer) to one node in the output layer and use the sigmoid transfer function (`sigmoid(x) = 1/(1+exp(-x))`).  This pattern is so common that any library for building neural networks will provide a function to build such a neural network easily.  Using Julia's Flux package, it's just `model_nn0 = Dense(6, 1, Flux.sigmoid)`.
"""

# ╔═╡ ddb12c2e-2764-432c-bc08-c056689e26fd
md"""
To train a neural network, we need to specify several key elements:
- parameters of the neural network:  
- loss function: The function to be minimized as a function of the parameters of the neural network.  
- training data: The observational data to be used by the loss function
- optimizer:  The algorithm that will be used for updating the parameters of the neural network.

Since neural networks can have *lots* of parameters, neural network libraries provide convenience functions to extract the parameters from a network.  We did that above with Flux's `params` function.  
"""

# ╔═╡ 5abe0169-d7e1-4fd2-b5b5-aae1a693572a
md"**Question:** How many model parameters will this neural network have?"

# ╔═╡ ec3a3d20-f7f0-491c-8b3a-69028ddf036f
md"""
!!! hint "Hint"
    sum(length.(nn_param)) = 7
"""

# ╔═╡ fdc82078-1a20-4a9b-84f0-d2010f258f03
md"""
As discussed in the logistic regression lab, the log likelihood for logistic regresssion is

$\ell(w)= \sum_{i=1}^n\left[y_i\log(h_w(x_i)) + (1-y_i)\log(1-h_w(x_i))\right],$

where $y_i$ are the labels for the $i$th object and $x_i$ are the input data for the $i$th object and $h_w(x_i)$ is the value of the output node, evaluated for the input data $x_i$.  The subscript $w$ emphasizes that the output of the neural network depends on a the values of the network's weights (and biases).

We estimate weights $w$ by the values that maximize the log-likelihood function above, or equivalently the values that minimize the negative log-likelihood function divided by the number of observations.  This is already coded up as the function `binarycrossentropy(predictions, observations)`.

!!! tip "Terminology"
    In the machine learning community, the function to be minimized is commonly called the loss function (or cost function).  This is slightly more general, since the loss function does not necessarily need to be proportional to a negative log likelihood.  For example, a loss function could also include a log prior term or a more arbitrary regularization penalty term.
"""

# ╔═╡ 014fb3c0-ae99-494a-8b2b-c02f4c6c6dd8
md"""
!!! tip "Pro tip:"
    We could get better numerical stability by defining the neural network to use a linear transfer function and replacing the loss function with `logitbinarycrossentropy`.  
"""

# ╔═╡ 12028b28-1547-40e6-8263-b32a80613914
md"""
It's natural to start using the gradient descent algorithm for optimizing our neural network parameters and a high learning rate of 0.9.  We could do that with `optimizer = Descent(0.9)`.  In practice, we've specified that the optimizer and its learning rate will be determined by your choice from a drop down box below.
"""

# ╔═╡ 83d2d3c5-f8ec-4bd6-9660-4dda9db75131
md"Now that all the elements are in place, you can visualize the results of training our neural network implementing logistic regression.  For starters, it defaults to using just 100 iterations (so the notebook responds quickly).  But you'll want to try continuing to train the network for more iterations to see if it can improve further.  
You can click `submit` below to continue training the neural network with specified learning rate and optimization algorithm.  Depending on the number of iterations it may take significant time to train the neural network and to update the plot below.  If the progress bar below is less than 100%, then please be patient while the neural network is training.  There's also a button you can click to reinitialize your neural network's weights (with random values)."

# ╔═╡ 0a8873a5-519a-4e6e-8b19-6fe6fc23fb1c
@bind reinit_nn Button("Reinitialize NN weights")

# ╔═╡ 7dccea44-ebe2-4882-8634-019c049591b2
begin
	reinit_nn  # trigger rerunning this cell when button is clicked
	# define neural network w/ 6 inputs, 1 output (and 0 hidden layers)
	model_nn0 = Dense(6, 1, sigmoid)  
	# extract the parameters of that neural network
	nn_param = Flux.params(model_nn0)  
end;

# ╔═╡ aca55f65-b364-4c20-8cc1-1dd0d4102ea1
loss(x,y) = Flux.binarycrossentropy(model_nn0(x), y)
#=  Or we could write the equivalent loss function ourselves using
function loss(x_obs,y_obs)
	ϵ = 1e-12  # To prevent issues with log(0)
	pred = model_nn0(x_obs)
	-mean( y_obs.*log.(pred.+ϵ) .+ (1 .-y_obs).*log.(1 .- pred .+ ϵ))
end
=#

# ╔═╡ ff3f98bf-d3b6-4937-86d1-47e0080a6568
md"""
**Question:**  How many iterations did you need to acheive a neural network classifier that performs comparably to the logistic regression classifier above?

**Question:**  How does the computational cost of training the neural network compare to the cost of fitting the logistic regression model?  What factors contribute to the difference?

**Question:**  How does the loss function change if you continue to train the neural network for many more iterations for the training dataset?  What about for the validation dataset?  What causes this difference?
"""

# ╔═╡ 69a9e752-bad1-4695-83b3-55dd5054c26f
md"""
!!! tip "How did we train the network?"
    If you're curious to see how the training was performed, you can read through the function `my_train!` near the end of the notebook.  Any modern neural network library will provide functions to help train your network (e.g., Flux's `train!`).  In this lab, we opted to use a handwritten function `my_train!` that is quite similar to `train!`, but hopefully is a little easier to read, so you can see what it's doing for yourself.
"""

# ╔═╡ 86c5f0d7-4fdd-4d99-af94-b0d3937580ac
md"""
## More General Neural Networks
In the section above, we intentionally used a very simple neural network that was equivalent to logistic regression.  
One of the features of using a neural network is that one can easily scale up the network to include more nodes and layers, so that it becomes a more flexible classifier.   

Below, you can select a number of hidden layers (up to 3) and number of hidden nodes in each layers.  Once you've selected and submitted an architecture and strategy for optimizing your network in the boxes below, click the checkbox below to start training it.  

!!! tip "Tip:"
    The time required to train neural networks can increase rapidly with their size, so start small.  Think about how many parameters (weights and bias terms) will be required for your neural network.  
"""

# ╔═╡ 1009975c-9a8b-40f6-b9c5-b69820adc6b1
md"Check once you're ready to proceed to neural networks using one or more hidden layers. $(@bind ready_for_hidden CheckBox())"

# ╔═╡ f5f1ff7c-c674-4147-9ec1-b1b0f1a8d18a
md"Number of hidden layers: $(@bind num_hidden_layers NumberField(1:3,default=1))"


# ╔═╡ 51218a3f-c7b9-4f8e-8b85-c2e8013cd13d
md"""
**Question:**  How does the number of iterations required to train the model well compare to the simpler neural network from the previous section?

**Question:**  How does the resulting classifier compare that of the simple neural network or the logistic regression classifier?  

**Question:**  Can you find a network architecture and training strategy that performs significantly better than the simple neural network and the logistic regression classifier?  What are its properties?  

**Question:**  Does your best neural network perform as well if you change the folds used for training and validation (way at the top of the notebook)?

**Question:**  Why was it so important to pick out a distinct test dataset at the very beginning of the notebook?
"""

# ╔═╡ afccede8-4563-4a7e-bca4-4754349e73b3
md"##  Setup & Helper functions"

# ╔═╡ 86744470-2b37-45c1-ab76-af838c122378
function find_or_download_data(data_filename::String, url::String)
	if contains(gethostname(),"ec2.internal")
		data_path = joinpath(homedir(),"data")
		isdir(data_path) || mkdir(data_path)
	elseif contains(gethostname(),"aci.ics.psu.edu")
		data_path = joinpath("/gpfs/scratch",ENV["USER"],"Astroinformatics")
		isdir(data_path) || mkdir(data_path)
		data_path = joinpath(data_path,"data")
		isdir(data_path) || mkdir(data_path)
	else
		data_path = joinpath(homedir(),"Astroinformatics")
		isdir(data_path) || mkdir(data_path)
		data_path = joinpath(data_path,"data")
		isdir(data_path) || mkdir(data_path)
	end
	data_path = joinpath(data_path,data_filename)
	if !(filesize(data_path) > 0)
		Downloads.download(url, data_path)
	end
	return data_path		
end


# ╔═╡ 49f371db-1ad1-4f1c-b5e2-c00a52035c6a
begin
	filename = "quasar2.csv"
	url = "https://scholarsphere.psu.edu/resources/edc61b33-550d-471d-8e86-1ff5cc8d8f4d/downloads/19732"
	data_path = find_or_download_data(filename, url);
end

# ╔═╡ 85066779-be0f-43a3-bde8-c4ab5a3e5ca3
begin
	df = CSV.read(data_path, DataFrame, limit=1_000_000, select=[:ug, :gr, :ri, :iz, :zs1, :s1s2, :label],  ntasks=1)
	df[:,:label] .= 1 .- df[:,:label]  # Make label=1 for high-z quasars
	col_names = names(df)
	df
end

# ╔═╡ f6e68394-34b6-453c-929f-7cc89434e179
base_rate_input = sum(df.label.==1)/length(df.label)

# ╔═╡ 0985bcc3-e686-4aa9-b832-0141cb27c4a4
begin
	frac_data_used_for_cv = 0.66
	df_cv, df_test = stratifiedobs(x->x.label==1, shuffleobs(df), p=frac_data_used_for_cv);
	if make_balanced
		df_cv = undersample(x->Bool(x.label),df_cv)
		df_test = undersample(x->Bool(x.label),df_test)
	end
end;

# ╔═╡ 798d218e-47d7-4c1a-bef4-8a93c0e52236
(;num_in_cv_set = size(df_cv,1), num_in_test_set = size(df_test,1) )

# ╔═╡ fecc9af4-7486-4d21-87bb-5ed1e35cdf6c
fm_logistic_regression = term("label") ~ term(1) + sum(term.(setdiff(col_names,["label"])));

# ╔═╡ ce34a578-ceaf-4c6e-aa73-9ab54cf3cf99
function stratified_kfolds(label::AbstractVector, data, num_folds::Integer)
	@assert length(label) == size(data,1)
	list_of_folds_idx = StratifiedKfold(label,num_folds)
	data_train = map(idx->datasubset(data, idx),list_of_folds_idx)
	data_test =  map(idx->datasubset(data, setdiff(1:length(label),idx)),
						list_of_folds_idx)
	(;data_train, data_test, folds_idx = list_of_folds_idx)
end

# ╔═╡ c41187d6-4306-4a92-b10e-c7825e79e79e
begin
	# For GLM models
	classify(model::RegressionModel, data::AbstractDataFrame; threshold::Real=0.5) = predict(model,data).>=threshold
	# For Flux models
	classify(model::Union{Chain,Dense}, data::AbstractMatrix; threshold::Real=0.5) = model(data).>=threshold
	classify(model::Union{Chain,Dense}, data::AbstractDataFrame; threshold::Real=0.5) = classify(model, Matrix(data)', threshold=threshold)'
end

# ╔═╡ b38f5368-b7a8-4c24-9d3b-e8159fa992da
accuracy(model, data::AbstractDataFrame, y::AbstractVector ) =
				sum(classify(model,data) .== Bool.(y) )/length(y)

# ╔═╡ 7d25013d-04f7-474f-8349-e623353c4fa0
function confusion_matrix_int01(a, b)
	@assert all(map(x->in(x,(0,1)),a))
	@assert all(map(x->in(x,(0,1)),b))
	transpose(hcat(.!Bool.(a), Bool.(a))) * hcat(.!Bool.(b), Bool.(b))
end

# ╔═╡ 7931116b-3b3f-455c-80aa-17de872a8965
function calc_classification_diagnostics(model, data, label; threshold = 0.5)
	pred = classify(model, data; threshold=threshold)
	num_true_positives  = sum(  label.==1 .&&   pred)
	num_true_negatives  = sum(  label.==0 .&& .!pred)
	num_false_negatives = sum(  label.==1 .&& .!pred)
	num_false_positives = sum(  label.==0 .&&   pred)

	num_condition_positives = num_true_positives + num_false_negatives
	num_condition_negatives = num_true_negatives + num_false_positives
	num_total = num_condition_positives + num_condition_negatives
	num_predicted_positives = num_true_positives + num_false_positives
	num_predicted_negatives = num_true_negatives + num_false_negatives
	true_positive_rate  = num_true_positives/num_condition_positives
	true_negative_rate  = num_true_negatives/num_condition_negatives
	false_positive_rate = num_false_positives/num_condition_negatives
	false_negative_rate = num_false_negatives/num_condition_positives
	accuracy = (num_true_positives+num_true_negatives)/num_total
	false_omission_rate = num_false_negatives / num_predicted_negatives
	false_discovery_rate = num_false_positives / num_predicted_positives
	F1_score = 2*num_true_positives/(2*num_true_positives+num_false_positives+num_false_negatives)
	prevalence = (num_true_positives+num_false_negatives)/num_total
	return (;threshold, accuracy, false_discovery_rate, false_omission_rate, F1_score,
		false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate,
		num_true_positives, num_true_negatives, num_false_positives, num_false_negatives,   
		num_condition_positives, num_condition_negatives, num_predicted_positives, num_predicted_negatives,
		num_total, prevalence )
end

# ╔═╡ 065cad81-4641-4517-a06c-817826e0fd9e
function run_cv_on_logistic_regression(df::AbstractDataFrame, folds_idx)
	@assert length(folds_idx)>=1
	@assert isa(first(folds_idx), AbstractVector{Int64})
	@assert length(first(folds_idx))>=1
	local history = MVHistory()
	# fold_idx comes from something like StratifiedKfold(df.label, num_folds )
	for (i,train_idx) in enumerate(folds_idx)
		validation_idx = setdiff(1:length(df.label),train_idx)
		model_logistic_regression = glm(fm_logistic_regression, df[train_idx,:], Binomial(), ProbitLink())

		result_train = calc_classification_diagnostics(model_logistic_regression, df[train_idx,:], df.label[train_idx])
		result_validation = calc_classification_diagnostics(model_logistic_regression, df[validation_idx,:], df.label[validation_idx])

		push!(history, :model, i, model_logistic_regression)
		push!(history, :loglikelihood, i, loglikelihood(model_logistic_regression))
		push!(history, :results_train, i, result_train)
		push!(history, :results_validation, i, result_validation)		
	end
	return history
end

# ╔═╡ 7179f3e7-7b8a-468b-847d-5962ce0c1a93
function my_train!(model_nn::Union{Dense,Chain}, loss::Function, param::Flux.Zygote.Params{PT},
				train_data::DT, optimizer::Flux.Optimise.AbstractOptimiser;
				#= begin optional parameters =#
				epochs::Integer=1, test_data = nothing) where { PT<:Any, MT1<:AbstractMatrix, MT2<:AbstractMatrix, DT<:Tuple{MT1,MT2} }
	@assert 1<=epochs<Inf
	if !isnothing(test_data)
		x_test, y_test = test_data
	end
	history = MVHistory()  # For storing intermediate results
	@progress for i in 1:epochs
		x, y = train_data
		results_train = calc_classification_diagnostics(model_nn, x, y)
		if !isnothing(test_data)  # if test/validation data is provied, evaluate model for it, too.
			results_test = calc_classification_diagnostics(model_nn, x_test, y_test)
			push!(history, :results_test, i, results_test)
			loss_test = loss(x_test, y_test)
			push!(history, :loss_test, i, loss_test )

		end
  		gs = gradient(param) do
		    loss(x,y)
	  	end
		push!(history, :loss, i, loss(x,y) )
		push!(history, :results_train, i, results_train)
		push!(history, :param, i, param)		

  		Flux.Optimise.update!(optimizer, param, gs)
	end
	return history
end

# ╔═╡ 5cbd25fc-557b-4578-b765-72fcd384d6e0
function plot_classifier_training_history(h::MVHistory, idx_plt)
	plt1 = plot(xlabel="Iteration", ylabel="False Discovery Rate", legend=:none)
	plt2 = plot(xlabel="Iteration", ylabel="False Omission Rate", legend=:none)
	plt3 = plot(xlabel="Iteration", ylabel="Loss", legend=:topright)
	scatter!(plt1,get(h,:results_train)[1][idx_plt],
					get.(get(h,:results_train)[2][idx_plt],
						:false_discovery_rate,nothing), 	ms=2, markerstrokewidth=0, alpha=0.5, label="Training")
	scatter!(plt2,get(h,:results_train)[1][idx_plt],
					get.(get(h,:results_train)[2][idx_plt],
						:false_omission_rate,nothing), 	ms=2, markerstrokewidth=0, alpha=0.5, label="Training")
	plot!(plt3,get(h,:loss)[1][idx_plt],
				get(h,:loss)[2][idx_plt],
				label="Training")
	if haskey(h,:results_test)
		scatter!(plt1,get(h,:results_test)[1][idx_plt],
						get.(get(h,:results_test)[2][idx_plt],
							:false_discovery_rate,nothing),
						ms=2, markerstrokewidth=0, alpha=0.5, label="Validation")
		scatter!(plt2,get(h,:results_test)[1][idx_plt],
						get.(get(h,:results_test)[2][idx_plt],
							:false_omission_rate,nothing),
						ms=2, markerstrokewidth=0, alpha=0.5, label="Validation")
		plot!(plt3,get(h,:loss_test)[1][idx_plt],
					get(h,:loss_test)[2][idx_plt],
						alpha=0.7, label="Validation")

	end
	l = @layout [ a;  b c]
	plot(plt3, plt1,plt2, layout=l)
end

# ╔═╡ 56529037-956d-4980-875e-85b0eb5644e0
TableOfContents()

# ╔═╡ 2bb65491-d291-4d71-ac5d-247538a1871b
nbsp = html"&nbsp;"

# ╔═╡ 608d9156-f8a8-4886-ae87-b1adab904de5
@bind param_fold confirm(PlutoUI.combine() do Child
md"""
Number of folds:  $(Child("num_folds", NumberField(1:10,default=5))) $nbsp $nbsp
Fold to use for validation:  $(Child("fold_id", NumberField(1:10)))
""" end )

# ╔═╡ ca25abcc-4f7e-4ace-861b-c8f0416584ed
if !(1<=param_fold.fold_id<=param_fold.num_folds)
	md"""
	!!! warn "fold_id must be between 1 and the number of folds"
	"""
else
	df_train_list, df_validation_list, list_of_folds_idx = stratified_kfolds(df_cv.label,df_cv,param_fold.num_folds)
	num_in_training_set = size(first(df_train_list),1)
	num_in_validation_set = size(first(df_validation_list),1)
	nothing
end

# ╔═╡ 73c4adf8-4c5e-4df5-9ba8-3099e79d3dcf
Markdown.parse("""
**Question:** Based on your choices above, approximately how many observations will be in each of the training datasets?  What about the validation datasets?

!!! hint "Answer:"
    Each training dataset has $num_in_training_set observations.
    Each validation dataset has $num_in_validation_set observations.
""")

# ╔═╡ 360bb5b4-ef16-4ba0-a4c0-dd4761a0e13d
lr_cv_results = run_cv_on_logistic_regression(df_cv, list_of_folds_idx);

# ╔═╡ 465c5c55-0d6d-444e-85fe-c40e4b79c462
let
	plt1 = plot(legend=:topleft)
	train_fdr = get.(get(lr_cv_results,:results_train)[2],:false_discovery_rate,nothing)
	valid_fdr = get.(get(lr_cv_results,:results_validation)[2],:false_discovery_rate,nothing)
	h_fdr = fit(Histogram, vcat(train_fdr,valid_fdr), nbins=20)

	histogram!(plt1,train_fdr, bins=h_fdr.edges, label="Train")
	histogram!(plt1,valid_fdr, bins=h_fdr.edges, label="Validation")
	xlabel!(plt1,"False Discovery Rate")
	xlims!(plt1,0,maximum(first(h_fdr.edges)))
	title!(plt1, "Logistic Regression Cross-validation results")

	plt2 = plot(legend=:topleft)
	train_for = get.(get(lr_cv_results,:results_train)[2],:false_omission_rate,nothing)
	valid_for = get.(get(lr_cv_results,:results_validation)[2],:false_omission_rate,nothing)
	h_for = fit(Histogram, vcat(train_for, valid_for), nbins=20)

	histogram!(plt2, train_for, bins=h_for.edges, label="Train")
	histogram!(plt2, valid_for, bins=h_for.edges, label="Validation")
	xlabel!(plt2,"False Omission Rate")
	xlims!(plt2,0,maximum(first(h_for.edges)))

	l = @layout [a;b]
	plot(plt1,plt2, layout=l)
end

# ╔═╡ 154a7a1d-170c-45a7-969a-83be985dbb0b
lr_cv_coeffs = hcat(coef.(get(lr_cv_results,:model)[2])...)

# ╔═╡ 324b3e47-15a0-4c5a-9d52-2f827c9fddd3
(;μ=mean(lr_cv_coeffs, dims=2) , σ=std(lr_cv_coeffs, dims=2) )

# ╔═╡ ddf9b625-3396-4866-b4ee-ac1cdd5b31f2
lr_cv_loss = -loglikelihood.(get(lr_cv_results,:model)[2])./length.(list_of_folds_idx)

# ╔═╡ 20965548-d105-4f59-ae1f-b08740c35de2
(;lr_cv_loss_mean = mean(lr_cv_loss), lr_cv_loss_std = std(lr_cv_loss) )

# ╔═╡ 6f2c856c-3bd3-4d35-be93-1b78c68c6b29
begin
	train_Xy, validation_Xy = df_train_list[param_fold.fold_id], df_validation_list[param_fold.fold_id]
	# Make some convenient variable names for use later
	train_X = select(train_Xy, Not(:label), copycols=false)
	train_y = select(train_Xy, (:label), copycols=false)
	validation_X = select(validation_Xy, Not(:label), copycols=false)
	validation_y = select(validation_Xy, (:label), copycols=false)
end;

# ╔═╡ 30090fcd-7c1b-4218-b7fa-d0d917cc13fe
model_logistic_regression = glm(fm_logistic_regression, train_Xy, Binomial(), ProbitLink())

# ╔═╡ d704ffca-e4c9-4b8f-ad4a-07e38fde820f
begin
	confusion_training = confusion_matrix_int01(train_y.label, classify(model_logistic_regression, train_X) ) ./length(train_y.label)

	confusion_validation = confusion_matrix_int01(validation_y.label, classify(model_logistic_regression, validation_X) ) ./length(validation_y.label)

	confusion_testing = confusion_matrix_int01(df_test.label, classify(model_logistic_regression, df_test) )./length(df_test.label)

	(;train=confusion_training, valid=confusion_validation, test=confusion_testing)
end

# ╔═╡ 0a736aaa-ba63-41ab-aab1-3f33f6cb7db0
@bind opt_param confirm(
	PlutoUI.combine() do Child
		md"""
		Learning Rate:  $( Child("learning_rate",NumberField(0.05:0.05:1, default=0.9)) )
		$nbsp $nbsp $nbsp
		Optimizer:  $( Child("type",Select([Descent => "Gradient Descent", Nesterov => 	"Nesterov Momentum", ADAM => "ADAM" ])) )


		Iterations:  $( Child("iterations",NumberField(10:10:2000, default=100)))
		$nbsp $nbsp $nbsp
		Compute validation data: $( Child("calc_validation_results", CheckBox()))
		"""
	end
	)

# ╔═╡ 776afa73-136a-4346-acaa-2a81a0d03728
optimizer = opt_param.type(opt_param.learning_rate)  # Set based on inputs above

# ╔═╡ d0124cc9-73e2-4374-b568-4c01c0396629
begin
	local train_data = (Matrix(train_X)', Matrix(train_y)')
	if opt_param.calc_validation_results
		local validation_data = (Matrix(validation_X)', Matrix(validation_y)')
		history_nn0 = my_train!(model_nn0, loss, nn_param, train_data, optimizer,
					test_data = validation_data , epochs=opt_param.iterations)
	else
		history_nn0 = my_train!(model_nn0, loss, nn_param, train_data, optimizer,
					epochs=opt_param.iterations)
	end
end;

# ╔═╡ 867f8614-794b-4fc7-90df-27037d28d56f
md"""
First iteration to plot: $(@bind first_iter_to_plot Slider(1:opt_param.iterations))
Last iteration to plot: $(@bind last_iter_to_plot Slider(1:opt_param.iterations,default=opt_param.iterations))
"""

# ╔═╡ e7617122-3141-404d-b429-8312d41e08ae
plot_classifier_training_history(history_nn0,first_iter_to_plot:last_iter_to_plot)

# ╔═╡ 9eea2ceb-94ee-4aa8-8f5e-ffdd4debe174
@bind num_nodes confirm(PlutoUI.combine() do Child
md"""
Nodes in each layer:

Hidden Layer 1: $(Child("hidden1",NumberField(1:10,default=2)))
$nbsp $nbsp
Hidden Layer 2: $(Child("hidden2",NumberField(0:(num_hidden_layers>=2 ? 10 : 0),default=num_hidden_layers>=2 ? 1 : 0)))
$nbsp $nbsp
Hidden Layer 3: $(Child("hidden3",NumberField(0:(num_hidden_layers>=3 ? 10 : 0),default=num_hidden_layers>=3 ? 1 : 0)))
"""
end)

# ╔═╡ 2af3988d-6a46-4ae6-ab77-9de5270bf657
md"Reinitialize neural network with hidden nodes: $nbsp $(@bind reinit_my_nn Button())"

# ╔═╡ c33ba176-e1bd-46a8-afca-a3d82eb4bc1a
if ready_for_hidden
	reinit_my_nn  # trigger rerunning this cell when button is clicked
	if num_hidden_layers == 1
		model_my_nn = Chain( Dense(6,num_nodes.hidden1, Flux.sigmoid),
						 Dense(num_nodes.hidden1, 1, Flux.sigmoid) )
	elseif num_hidden_layers == 2
		model_my_nn = Chain( Dense(6,num_nodes.hidden1, Flux.sigmoid),
						 Dense(num_nodes.hidden1, num_nodes.hidden2, Flux.sigmoid),
	 					 Dense(num_nodes.hidden2, 1, Flux.sigmoid) )
	elseif num_hidden_layers == 3
		model_my_nn = Chain( Dense(6,num_nodes.hidden1, Flux.sigmoid),
						 Dense(num_nodes.hidden1, num_nodes.hidden2, Flux.sigmoid),
						 Dense(num_nodes.hidden2, num_nodes.hidden3, Flux.sigmoid),
	 					 Dense(num_nodes.hidden3, 1, Flux.sigmoid) )
	else
		md"""!!! warn "Invalid number of layers"""
	end
	my_nn_param = Flux.params(model_my_nn)  
	my_loss(x,y) = Flux.binarycrossentropy(model_my_nn(x), y)
end;

# ╔═╡ 24740804-7333-4e93-aff9-badede5c440c
if ready_for_hidden
	num_param_in_my_nn = sum(length.(my_nn_param))
	md"The new neural network with $num_hidden_layers has $num_param_in_my_nn parameters."
end

# ╔═╡ c799d55a-2fb9-4b0a-8ebf-12f9cd4b95db
begin
	@bind my_opt_param confirm(
	PlutoUI.combine() do Child
		md"""
		Learning Rate:  $( Child("learning_rate",NumberField(0.05:0.05:1, default=0.9)) )
		$nbsp $nbsp $nbsp
		Optimizer:  $( Child("type",Select([Descent => "Gradient Descent", Nesterov => 	"Nesterov Momentum", ADAM => "ADAM" ], default=Nesterov)) )
		$nbsp $nbsp $nbsp
		Iterations:  $( Child("iterations",NumberField(10:10:2000, default=500)))
		"""
	end
	)
end

# ╔═╡ a4e39577-39ff-4295-a345-c580a062ad01
if ready_for_hidden  
	my_optimizer = my_opt_param.type(my_opt_param.learning_rate)  # Set based on inputs above
end;

# ╔═╡ 7369a73e-a04f-49c9-83f8-82633f8c3efb
if ready_for_hidden
	local train_data = (Matrix(train_X)', Matrix(train_y)')
	local validation_data = (Matrix(validation_X)', Matrix(validation_y)')

	history_my_nn = my_train!(model_my_nn, my_loss, my_nn_param, train_data, my_optimizer,
					test_data = validation_data, epochs=my_opt_param.iterations)
end;

# ╔═╡ 86499d0e-bad3-4954-a740-68cba383d790
if ready_for_hidden
md"""
First iteration to plot: $(@bind first_iter_to_plot_hidden Slider(1:my_opt_param.iterations))
Last iteration to plot: $(@bind last_iter_to_plot_hidden Slider(1:my_opt_param.iterations,default=my_opt_param.iterations))
"""
end

# ╔═╡ 9b55d85d-e5b0-46d6-bce8-6f1cbdd991ee
if ready_for_hidden
	plot_classifier_training_history(history_my_nn,first_iter_to_plot_hidden:last_iter_to_plot_hidden)
end

# ╔═╡ 9c5a7bb8-2017-45e5-b56e-3745fc775e7c
br = html"<br />"

# ╔═╡ 1d8fe699-810b-44ba-a9cd-80816338e08c
md"""
# Lab 14: Neural Networks for $br Identifying high-z quasars
#### [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
#### [Eric Ford](https://www.personal.psu.edu/ebf11)
"""

# ╔═╡ 3a1d0d17-b17b-4a6a-b997-f3924211ea2d
md"### Old code"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
MLDataUtils = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
ValueHistories = "98cad3c8-aec3-5f06-8e41-884608649ab7"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.4"
Flux = "~0.13.0"
GLM = "~1.7.0"
MLBase = "~0.9.0"
MLDataUtils = "~0.5.4"
Plots = "~1.29.0"
PlutoUI = "~0.7.38"
ProgressLogging = "~0.1.4"
StatsBase = "~0.33.16"
ValueHistories = "~0.5.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Future", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "0264a938934447408c7f0be8985afec2a2237af4"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.11"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "81f0cb60dc994ca17f68d9fb7c942a5ae70d9ee4"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.8"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "19fb33957a5f85efb3cc10e70cf4dd4e30174ac9"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.10.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "de68815ccf15c7d3e3e3338f0bd3a8a0528f9b9f"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.33.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a985dc37e357a3b22b260a5def99f3530fb415d3"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.2"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "8ccaa8c655bc1b83d2da4d569c9b28254ababd6e"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.2"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "8a6b49396a4058771c5c072239b2e0a76e2e898c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.58"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "4391d3ed58db9dc5a9883b23a0578316b4798b1f"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.0"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "f84e50845ab88702c721dc7c6129a85cbc1de332"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.1"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "92b8d38886445d6d06e5f13201e57d018c4ff880"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.7.0"

[[deps.GPUArrays]]
deps = ["Adapt", "LLVM", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "c783e8883028bf26fb05ed4022c450ef44edd875"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.3.2"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "d8c5999631e1dc18d767883f621639c838f8e632"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.15.2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b316fd18f5bc025fedcb708332aecb3e13b9b453"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1e5490a51b4e9d07e8b04836f6008f46b48aaa87"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.3+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "c8d47589611803a0f3b4813d9e267cd4e3dbcefb"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.11.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LearnBase]]
deps = ["LinearAlgebra", "StatsBase"]
git-tree-sha1 = "47e6f4623c1db88570c7a7fa66c6528b92ba4725"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase"]
git-tree-sha1 = "3bd9fd4baf19dfc1edf344bc578da7f565da2e18"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.9.0"

[[deps.MLDataPattern]]
deps = ["LearnBase", "MLLabelUtils", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e99514e96e8b8129bb333c69e063a56ab6402b5b"
uuid = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
version = "0.5.4"

[[deps.MLDataUtils]]
deps = ["DataFrames", "DelimitedFiles", "LearnBase", "MLDataPattern", "MLLabelUtils", "Statistics", "StatsBase"]
git-tree-sha1 = "ee54803aea12b9c8ee972e78ece11ac6023715e6"
uuid = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
version = "0.5.4"

[[deps.MLLabelUtils]]
deps = ["LearnBase", "MappedArrays", "StatsBase"]
git-tree-sha1 = "fd75d4b0c4016e047bbb6263eecf7ae3891af522"
uuid = "66a33bbf-0c2b-5fc8-a008-9da813334f0a"
version = "0.5.7"

[[deps.MLStyle]]
git-tree-sha1 = "e49789e5eb7b2d5577aaea395bfcac769df64bb8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.11"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase"]
git-tree-sha1 = "95ab49a8c9afb6a8a0fc81df25617a6798c0fb73"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.5"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "f89de462a7bc3243f95834e75751d70b3a33e59d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.5"

[[deps.NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "0d18b4c80a92a00d3d96e8f9677511a7422a946e"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.2"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2442c3ddbda547c80e8b6451a103719d6a3593dd"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.4"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "afeacaecf4ed1649555a19cb2cad3c141bbc9474"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.5.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "3a2a99b067090deb096edecec1dc291c5b4b31cb"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "4352d5badd1bc8bf0a8c825e886fa1eda4f0f967"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.30"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7638550aaea1c9a1e86817a231ef0faa9aca79bd"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.19"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.ValueHistories]]
deps = ["DataStructures", "RecipesBase"]
git-tree-sha1 = "9cc583107e0c09b8986ac10660653fa2e37c0f62"
uuid = "98cad3c8-aec3-5f06-8e41-884608649ab7"
version = "0.5.4"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a49267a2e5f113c7afe93843deea7461c0f6b206"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.40"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─1d8fe699-810b-44ba-a9cd-80816338e08c
# ╟─2671099c-667e-4c1b-9e5d-f41bd5752938
# ╟─783788a0-6f9a-4225-98e6-5030d9f21712
# ╠═49f371db-1ad1-4f1c-b5e2-c00a52035c6a
# ╠═85066779-be0f-43a3-bde8-c4ab5a3e5ca3
# ╠═f6e68394-34b6-453c-929f-7cc89434e179
# ╟─1c792c1d-f1e8-4e5f-8a76-5c7ca5fb8587
# ╟─26635f45-1e34-4025-8151-2185d8d84e06
# ╠═0985bcc3-e686-4aa9-b832-0141cb27c4a4
# ╟─798d218e-47d7-4c1a-bef4-8a93c0e52236
# ╟─ffe4cdc7-4863-4f0e-b790-4b86afcc56b8
# ╟─608d9156-f8a8-4886-ae87-b1adab904de5
# ╟─ca25abcc-4f7e-4ace-861b-c8f0416584ed
# ╟─73c4adf8-4c5e-4df5-9ba8-3099e79d3dcf
# ╟─7e0341a3-e52e-4a19-b9ac-969ebdd2161f
# ╠═6f2c856c-3bd3-4d35-be93-1b78c68c6b29
# ╟─4aff0575-014d-4757-8e69-f49ad1f2e82e
# ╠═fecc9af4-7486-4d21-87bb-5ed1e35cdf6c
# ╟─65472ed7-4c0a-4695-8dc3-94575a0241dd
# ╠═30090fcd-7c1b-4218-b7fa-d0d917cc13fe
# ╟─b419a260-8ec8-46f9-bc5b-0fdd09717867
# ╟─d704ffca-e4c9-4b8f-ad4a-07e38fde820f
# ╟─6636da1a-64c0-4c12-b102-82a5d49df114
# ╟─7022d21e-10f3-46ac-a5fa-b4b6f7ab7881
# ╠═360bb5b4-ef16-4ba0-a4c0-dd4761a0e13d
# ╟─465c5c55-0d6d-444e-85fe-c40e4b79c462
# ╠═154a7a1d-170c-45a7-969a-83be985dbb0b
# ╟─f1e1d4ab-d1bf-4684-b3f6-3233b689ba71
# ╠═324b3e47-15a0-4c5a-9d52-2f827c9fddd3
# ╟─54f52161-2337-40fc-a743-58cc0492726f
# ╟─84c733a4-5f23-42d6-bea5-c7cc89674d41
# ╠═ddf9b625-3396-4866-b4ee-ac1cdd5b31f2
# ╟─20965548-d105-4f59-ae1f-b08740c35de2
# ╟─bd6b7f5f-7f64-41ce-9ce8-639d6435800a
# ╠═7dccea44-ebe2-4882-8634-019c049591b2
# ╟─ddb12c2e-2764-432c-bc08-c056689e26fd
# ╟─5abe0169-d7e1-4fd2-b5b5-aae1a693572a
# ╟─ec3a3d20-f7f0-491c-8b3a-69028ddf036f
# ╟─fdc82078-1a20-4a9b-84f0-d2010f258f03
# ╠═aca55f65-b364-4c20-8cc1-1dd0d4102ea1
# ╟─014fb3c0-ae99-494a-8b2b-c02f4c6c6dd8
# ╟─12028b28-1547-40e6-8263-b32a80613914
# ╠═776afa73-136a-4346-acaa-2a81a0d03728
# ╟─83d2d3c5-f8ec-4bd6-9660-4dda9db75131
# ╟─0a8873a5-519a-4e6e-8b19-6fe6fc23fb1c
# ╟─0a736aaa-ba63-41ab-aab1-3f33f6cb7db0
# ╟─d0124cc9-73e2-4374-b568-4c01c0396629
# ╠═e7617122-3141-404d-b429-8312d41e08ae
# ╟─867f8614-794b-4fc7-90df-27037d28d56f
# ╟─ff3f98bf-d3b6-4937-86d1-47e0080a6568
# ╟─69a9e752-bad1-4695-83b3-55dd5054c26f
# ╟─86c5f0d7-4fdd-4d99-af94-b0d3937580ac
# ╟─1009975c-9a8b-40f6-b9c5-b69820adc6b1
# ╟─f5f1ff7c-c674-4147-9ec1-b1b0f1a8d18a
# ╟─9eea2ceb-94ee-4aa8-8f5e-ffdd4debe174
# ╟─c33ba176-e1bd-46a8-afca-a3d82eb4bc1a
# ╟─24740804-7333-4e93-aff9-badede5c440c
# ╟─2af3988d-6a46-4ae6-ab77-9de5270bf657
# ╟─c799d55a-2fb9-4b0a-8ebf-12f9cd4b95db
# ╟─a4e39577-39ff-4295-a345-c580a062ad01
# ╟─7369a73e-a04f-49c9-83f8-82633f8c3efb
# ╟─9b55d85d-e5b0-46d6-bce8-6f1cbdd991ee
# ╟─86499d0e-bad3-4954-a740-68cba383d790
# ╟─51218a3f-c7b9-4f8e-8b85-c2e8013cd13d
# ╟─afccede8-4563-4a7e-bca4-4754349e73b3
# ╠═423cb435-ab09-426e-a29a-0b894fc767ba
# ╠═065cad81-4641-4517-a06c-817826e0fd9e
# ╠═86744470-2b37-45c1-ab76-af838c122378
# ╠═ce34a578-ceaf-4c6e-aa73-9ab54cf3cf99
# ╠═c41187d6-4306-4a92-b10e-c7825e79e79e
# ╟─b38f5368-b7a8-4c24-9d3b-e8159fa992da
# ╟─7d25013d-04f7-474f-8349-e623353c4fa0
# ╟─7931116b-3b3f-455c-80aa-17de872a8965
# ╠═7179f3e7-7b8a-468b-847d-5962ce0c1a93
# ╠═5cbd25fc-557b-4578-b765-72fcd384d6e0
# ╠═56529037-956d-4980-875e-85b0eb5644e0
# ╟─2bb65491-d291-4d71-ac5d-247538a1871b
# ╟─9c5a7bb8-2017-45e5-b56e-3745fc775e7c
# ╟─3a1d0d17-b17b-4a6a-b997-f3924211ea2d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
