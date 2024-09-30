#install.packages("MXM", repos = "http://cran.us.r-project.org")
library("MXM")

# arguments
args = commandArgs(trailingOnly=TRUE)
dataset_name <- args[1]
target_name <- args[2]
ind_test_name <- args[3]
alpha <- as.double(args[4])
k <- strtoi(args[5])
output_file <- args[6]
verbose <- as.logical(args[7])

if (length(args) == 8) {
    train_idx_name <- args[8]
}

dataset <- read.csv(dataset_name, header=TRUE)
dataset <- as.matrix(sapply(dataset, as.numeric))

if (length(args) == 8) {
    train_idx <- read.csv(train_idx_name, header=TRUE)
    r_train_idx <- train_idx$train_idx + 1
    train_data <- dataset[r_train_idx,]
} else {
    train_data <- dataset
}

target_data <- train_data[, target_name]
feature_data <- train_data[, colnames(dataset) != target_name]

if (verbose) {
    cat("Arguments:\n")
    cat("dataset_name: ", dataset_name, "\n")
    cat("target_name: ", target_name, "\n")
    cat("ind_test_name: ", ind_test_name, "\n")
    cat("alpha: ", alpha, "\n")
    cat("k: ", k, "\n")
    cat("output_file: ", output_file, "\n")
    if (length(args) == 8) {
        cat("train_idx_name: ", train_idx_name, "\n")
    }
    cat("Dataset loaded:\n")
    print(head(dataset))
    cat("Train data:\n")
    print(head(train_data))
    cat("Target data:\n")
    print(head(target_data))
    cat("Feature data:\n")
    print(head(feature_data))
}

ses_object <- SES(target_data, feature_data, threshold = alpha, test = ind_test_name, max_k = k)
selectedVars <- ses_object@selectedVars

# Adjust the selectedVars to match the original dataset indexes
original_columns <- colnames(dataset)
feature_columns <- original_columns[original_columns != target_name]
selected_feature_indexes <- match(feature_columns[selectedVars], original_columns)

# Convert to data.frame and rename the column to 'sel'
selectedVars_df <- data.frame(sel = selected_feature_indexes - 1)  # Subtract 1 to match Python indexing
print(selectedVars_df)
write.csv(selectedVars_df, output_file, row.names = FALSE)
