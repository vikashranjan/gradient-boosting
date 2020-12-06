using Printf, Statistics

struct Dataset
    X::Array
    y::Array
end

mutable struct Node
    is_leaf::Bool
    left_child::Node
    right_child::Node
    split_feature::Int32
    split_value::Float64    
    weight::Float64
    Node()=new()
end

mutable struct Tree
    root::Node
    Tree()=new()
end

macro assert(ex)
    return :( $ex ? nothing : throw(AssertionError($(string(ex)))) )
end

function cal_term(g::Float64,h::Float64,lambda::Float64)
    return g^2/(h+lambda)
end

function split_gain(G::Float64, H::Float64, G_l::Float64, H_l::Float64, G_r::Float64, H_r::Float64, lambd::Float64)
    return cal_term(G_l,H_l,lambd)+cal_term(G_r,H_r,lambd)-cal_term(G,H,lambd)
end

function leaf_weight(G::Float64, H::Float64, lambd::Float64)
    return G/(H+lambd)
end

function len(X::Array)
    return size(X)[1]
end
function dim(X::Array)
    return size(X)[2]
end

function predict(x::Array,n::Node)
if n.is_leaf
    return n.weight
else
   
    if x[n.split_feature] <= n.split_value
        return predict(x,n.left_child)
    else
        return predict(x,n.right_child)
    end
end
end

function build_tree!(t::Tree, X::Array,grad::Array,hessian::Array,shrinkage_rate::Float64,param::Dict)
    #@assert len(X) == len(grad) == len(hessian)
    t.root = Node()
    current_depth = 0
    build_node!(t.root,X, grad, hessian, shrinkage_rate, current_depth, param)
end

function predict(x,t::Tree)
    return predict(x,t.root)
end

function build_node!(n::Node,X::Array,grad::Array,hessian::Array,shrinkage_rate::Float64,depth::Int,param::Dict)
    G=sum(grad)
    H=sum(hessian)
    lambda=param["lambda"]
    max_depth=param["max_depth"]
    min_split_gain=param["min_split_gain"]
    if depth > max_depth
        n.is_leaf=true
        n.weight = leaf_weight(G,H,lambda)* shrinkage_rate
        return
    end

    best_gain = 0.
    best_feature_id = ""
    best_val = 0.
    best_left_instance_ids = ""
    best_right_instance_ids = ""

    #find the best split
    for feature_id in 1:dim(X)
        G_l, H_l = 0., 0.
        sorted_instance_ids = sortperm(X[:,feature_id])
        for j in 1:len(sorted_instance_ids)-1
            G_l += grad[sorted_instance_ids[j]]
            H_l += hessian[sorted_instance_ids[j]]
            G_r = G - G_l
            H_r = H - H_l
            current_gain = split_gain(G, H, G_l, H_l, G_r, H_r, lambda)
            if current_gain > best_gain
                best_gain = current_gain
                best_feature_id = feature_id
                # print(sorted_instance_ids[j],",",feature_id,"\n")
                best_val = X[sorted_instance_ids[j],:][feature_id]
                best_left_instance_ids = sorted_instance_ids[begin:j+1]
                best_right_instance_ids = sorted_instance_ids[j+1:end]
            end
        end
    end


    if best_gain < min_split_gain
        n.is_leaf = true
        n.weight =  leaf_weight(G, H, lambda) * shrinkage_rate
    else
        n.split_feature = best_feature_id
        n.split_value = best_val

        n.left_child = Node()
        build_node!(n.left_child,X[best_left_instance_ids,:],
                              grad[best_left_instance_ids,:],
                              hessian[best_left_instance_ids,:],
                              shrinkage_rate,
                              depth+1, param)

        n.right_child = Node()
        build_node!(n.right_child,X[best_right_instance_ids,:],
                               grad[best_right_instance_ids,:],
                               hessian[best_right_instance_ids,:],
                               shrinkage_rate,
                               depth+1, param)
    end
end

mutable struct GradientBoostedTree
    params::Dict
    best_iteration::Int
    models::Array{Tree}
    GradientBoostedTree()=new()
end

function predict(bt::GradientBoostedTree, x)
    return predict(x,bt.models[1:bt.best_iteration])
end
function predict(x, models::Array{Tree})
    if len(models)!=0
        return sum([predict(x,m) for m in models])
    else
        return 0
    end
end
function training_data_scores(train_set::Dataset, models::Array)
    X = train_set.X
    scores = zeros(len(X))

    if len(models) == 0
        return  scores
    end
    
    for i in 1:len(X)
        scores[i] = predict(X[i,:], models)
    end
return scores
end

function calc_l2_gradient(train_set, scores)
    labels = train_set.y
    hessian = ones(len(labels))*.2
    
    grad = [2 * (labels[i] - scores[i]) for i in 1:len(labels)]
return grad, hessian
end

function calc_gradient(train_set, scores)
    return calc_l2_gradient(train_set, scores)
end

function calc_l2_loss(models, data_set)
    N=len(data_set.X)
    errors=zeros(N)
    for i in 1:len(data_set.X)
        errors[i]=(data_set.y[i]-predict(data_set.X[i,:], models))
    end
    return mean(errors.^2)
end

function calc_loss(models, data_set)
    return calc_l2_loss(models, data_set)
end

function build_gbt(bt::GradientBoostedTree,train_set, grad, hessian, shrinkage_rate)
    learner = Tree()
    build_tree!(learner,train_set.X, grad, hessian, shrinkage_rate, bt.params)
    return learner
end

function train!(bt::GradientBoostedTree, params::Dict, train_set::Dataset, num_boost_round::Int, valid_set::Dataset, early_stopping_rounds::Int)
    bt.params = params
    models = Tree[]
    shrinkage_rate = 1.
    best_iteration = 0
    best_val_loss = typemax(Float64)
    train_start_time = time()
    @printf("Training until validation scores don't improve for %i rounds. \n",early_stopping_rounds)

for iter_cnt in 1:num_boost_round
    iter_start_time = time()
    scores = training_data_scores(train_set, models)
    grad, hessian = calc_gradient(train_set, scores)
    learner = build_gbt(bt,train_set, grad, hessian, shrinkage_rate)

    if iter_cnt > 0
        shrinkage_rate *= bt.params["learning_rate"]
    end
    push!(models,learner)

    train_loss =calc_loss(models, train_set)
    val_loss = calc_loss(models, valid_set) 
 
    @printf("Iter %i, Train's L2: %f, Valid's L2: %f, Elapsed: %f secs \n",iter_cnt, train_loss, val_loss, time() - iter_start_time)

    if val_loss < best_val_loss
        best_val_loss = val_loss
        best_iteration = iter_cnt
    end
    if iter_cnt - best_iteration >= early_stopping_rounds
        print("Early stopping, best iteration is: \n")
        @printf("Iter %f, Train's L2: %f",best_iteration, best_val_loss)
        break
    end
end
    bt.models = models
    bt.best_iteration = best_iteration

    @printf("Training finished. Elapsed: %f secs",(time() - train_start_time))
end

function rmse(y,ypred)
    N=len(y)
    return sqrt(sum((y_test.-y_pred).^2)/N)
end