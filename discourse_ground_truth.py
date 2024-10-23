import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from german_generation import save_dataframe


def load_data():

    df_discourse=pd.read_csv('discourse_full_context_semantic.csv')

    #FULL CONTEXT (=1024)

    ground_truth=df_discourse.copy()
    ground_truth=preprocessing_data(ground_truth)

    generated_text = pd.read_csv(get_most_recent_csv('german_metrics_mean_and_sd'))

    return ground_truth, generated_text



def get_most_recent_csv(prefix):

    directory = "/home/ubuntu/helene/results"

    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out only CSV files
    csv_files = [f for f in files if f.endswith('.csv') and f.startswith(prefix)] 
    
    # Get the full path of the files
    full_paths = [os.path.join(directory, f) for f in csv_files]
    
    # Sort files by modification time in descending order
    most_recent_file = max(full_paths, key=os.path.getmtime)
    
    return most_recent_file



def preprocessing_data(df):
    # Columns to drop
    columns_to_drop = [
        'Unnamed: 0', 'task', 'context', 'log_probabilities_sd',
        'log_probabilities_mean', 'token_probabilities_sd',
        'token_probabilities_mean', 'panss_p_total', 'panss_n_total',
        'panss_g_total', 'panss_sim_total', 'tlc_tat_total', 'tlc_panss_total',
        'tlc_conversation_total', 'tlc_panss_global', 'tlc_conversation_global',
        'tlc_tat_global'
    ]

    # Drop the specified columns
    df_cleaned = df.drop(columns=columns_to_drop)
    print(df_cleaned.columns)

    df_cleaned = df_cleaned[~df_cleaned['sub_task'].isin([4,5,7])]
    df_cleaned = df_cleaned[~df_cleaned['sub_task'].isin([3,6])] #just for now for my short generated file that has only prompt 3 and 6
    print(df_cleaned['sub_task'])

    return df_cleaned


#correlation
def correlation_matrix(df_cleaned):

    # Calculate correlation matrix for means
    corr_matrix_means = df_cleaned[['probability_differences_mean', 'entropies_mean',
                            'information_content_mean', 
                            'entropy_deviations_mean',
                            'perplexity' 
                            ]].corr() 

    # Plot the heatmap for means
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_means, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Mean Metrics')
    plt.show()

    return corr_matrix_means
    

def drop_most_correlated_metric(df_cleaned):
    corr_matrix_means = correlation_matrix(df_cleaned)
    
    # Calculate the sum of absolute correlations for each metric
    abs_corr_sums = corr_matrix_means.abs().sum() - 1  # Subtract 1 to exclude self-correlation
    
    # Identify the metric with the highest total absolute correlation
    metric_to_drop = abs_corr_sums.idxmax()
    
    # Extract the prefix of the metric to be dropped (e.g., 'entropies')
    metric_prefix = metric_to_drop.rsplit('_', 1)[0]
    
    # Drop the mean and standard deviation columns for the identified metric
    columns_to_drop = [col for col in df_cleaned.columns if col.startswith(metric_prefix)]
    df_dropped = df_cleaned.drop(columns=columns_to_drop)
    
    print(f"Dropped columns: {columns_to_drop}")
    
    return df_dropped


# Bhattacharyya Distance
# Applicability: It is specifically designed to measure the similarity between two probability distributions based on their mean and variance (or standard deviation).
# Characteristics:
# Considers both the mean and standard deviation of the distributions.
# It is symmetric and ranges from 0 (identical distributions) to positive infinity.
# Particularly useful when comparing Gaussian distributions (since mean and standard deviation fully characterize Gaussian distributions).

#Compare generated text with real data

import math

def bhattacharyya_distance(mean1, sd1, mean2, sd2, epsilon=1e-8):
    """Calculate the Bhattacharyya distance between two distributions defined by mean and standard deviation."""
    # sd1 += epsilon
    # sd2 += epsilon
    term1 = 0.25 * np.log(0.25 * ((sd1/sd2)**2 + (sd2/sd1)**2 + 2))
    term2 = 0.25 * (((mean1 - mean2)**2) / (sd1**2 + sd2**2))
    return term1 + term2


def calculate_sequence_distance(seq1, seq2):
    """Calculate the total distance between two sequences based on the given metrics."""
    metrics = ['entropies', 'entropy_deviations', 'probability_differences']
    
    total_distance = 0
    for metric in metrics:
        mean1 = seq1[f'{metric}_mean']
        sd1 = seq1[f'{metric}_sd']
        mean2 = seq2[f'{metric}_mean']
        sd2 = seq2[f'{metric}_sd']
        dist = bhattacharyya_distance(mean1, sd1, mean2, sd2)
        total_distance += dist
    
    # Calculate distance for perplexity directly
    perplexity_diff = abs(seq1['perplexity'] - seq2['perplexity'])
    total_distance += perplexity_diff
    
    return total_distance


def calculate_group_distance(gt_group, gen_group):
    """Calculate the total distance between two groups of sequences."""
    total_distance = 0
    sub_tasks_prompts = [1, 2] #, 3, 6] to change when right document has been generated
    
    for number in sub_tasks_prompts:

        if number not in gt_group['sub_task'].values:
            print(f"Sub_task {number} not found in gt_group. {gt_group}")
            continue
        
        # Check if the prompt exists in gen_group
        if number not in gen_group['prompt'].values:
            print(f"Prompt {number} not found in gen_group.")
            continue

        seq1 = gt_group[gt_group['sub_task']==number].iloc[0]
        seq2 = gen_group[gen_group['prompt']==number].iloc[0]
        # print(f'seq1 : {seq1} and seq2: {seq2}')
        total_distance += calculate_sequence_distance(seq1, seq2)
    
    return total_distance


# To group by category of patient

# def calculate_distances(ground_truth_df, generated_text_df):
#     """Calculate the distance matrix between grouped ground_truth and generated_text sequences."""
#     ground_truth_groups = ground_truth_df.groupby('group')  # Group by 'group' instead of 'study_id'
#     generated_text_groups = generated_text_df.groupby(['sampling_method', 'p', 'temperature', 'backward_attention_length'])
#     print(f'ground truth groups {len(ground_truth_groups)} and {ground_truth_groups.head()}')
#     distances = []
    
#     for gt_group_label, gt_group in ground_truth_groups:
#         # Iterate over the groups in generated text
#         for (gen_sampling_method, gen_p, gen_temperature, gen_backward_length), gen_group in generated_text_groups:
#             distance = calculate_group_distance(gt_group, gen_group)
            
#             distances.append({
#                 'gt_group': gt_group_label,  # Store the group label instead of study_id
#                 'gen_sampling_method': gen_sampling_method,
#                 'gen_p': gen_p,
#                 'gen_temperature': gen_temperature,
#                 'gen_backward_length': gen_backward_length,
#                 'distance': distance
#             })
    
#     distances_df = pd.DataFrame(distances)
#     return distances_df


# def optimize_assignment(distance_df, ground_truth_df):
#     """Optimize the assignment of ground_truth groups to generated_text groups."""
#     # Pivot table now uses 'gt_group' instead of 'gt_study_id'
#     distances_pivot = distance_df.pivot_table(index='gt_group', columns=['gen_sampling_method', 'gen_p', 'gen_temperature', 'gen_backward_length'], values='distance')
#     distances_matrix = distances_pivot.values

#     # Check for NaNs or infinite values and replace them with a large number
#     if np.any(np.isnan(distances_matrix)):
#         print("NaN values found in the distance matrix. Replacing with a large number.")
#         distances_matrix = np.nan_to_num(distances_matrix, nan=np.inf)

#     if np.any(np.isinf(distances_matrix)):
#         print("Infinite values found in the distance matrix. Adjusting them.")
#         distances_matrix[distances_matrix == np.inf] = np.max(distances_matrix[np.isfinite(distances_matrix)]) * 1e6

#     try:
#         row_ind, col_ind = linear_sum_assignment(distances_matrix)
#     except ValueError as e:
#         print(f"Error during linear sum assignment: {e}")
#         return pd.DataFrame()  # Return an empty dataframe if assignment fails

#     assignments = []
    
#     for i, gt_index in enumerate(row_ind):
#         gt_group = distances_pivot.index[gt_index]
#         gen_sampling_method, gen_p, gen_temperature, gen_backward_length = distances_pivot.columns[col_ind[i]]
#         distance = distances_matrix[gt_index, col_ind[i]]
        
#         assignments.append({
#             'gt_group': gt_group,
#             'gen_sampling_method': gen_sampling_method,
#             'gen_p': gen_p,
#             'gen_temperature': gen_temperature,
#             'gen_backward_length': gen_backward_length,
#             'distance': distance
#         })
    
#     assignments_df = pd.DataFrame(assignments)
#     save_dataframe(assignments_df, 'assignments')
    
#     return assignments_df



# To group by study_id

def calculate_distances(ground_truth_df, generated_text_df):
    """Calculate the distance matrix between grouped ground_truth and generated_text sequences."""
    ground_truth_groups = ground_truth_df.groupby('study_id')
    generated_text_groups = generated_text_df.groupby(['sampling_method', 'p', 'temperature', 'backward_attention_length'])
    
    distances = []
    
    for gt_study_id, gt_group in ground_truth_groups:
    #     if len(gt_group['sub_task'].unique()) != 4:
    #         continue  # Skip groups that do not have all 4 sub_tasks
        
        for (gen_sampling_method, gen_p, gen_temperature, gen_backward_length), gen_group in generated_text_groups:
            # if len(gen_group['prompt'].unique()) != 4:
            #     continue  # Skip groups that do not have all 4 prompts
            
            distance = calculate_group_distance(gt_group, gen_group)
            
            distances.append({
                'gt_study_id': gt_study_id,
                'gen_sampling_method': gen_sampling_method,
                'gen_p': gen_p,
                'gen_temperature': gen_temperature,
                'gen_backward_length': gen_backward_length,
                'distance': distance
            })
    
    distances_df = pd.DataFrame(distances)
    
    return distances_df



def optimize_assignment(distance_df, ground_truth_df):
    """Optimize the assignment of ground_truth groups to generated_text groups."""
    distances_pivot = distance_df.pivot_table(index='gt_study_id', columns=['gen_sampling_method', 'gen_p', 'gen_temperature', 'gen_backward_length'], values='distance')
    distances_matrix = distances_pivot.values

     # Check for NaNs or infinite values and replace them with a large number
    if np.any(np.isnan(distances_matrix)):
        print("NaN values found in the distance matrix. Replacing with a large number.")
        distances_matrix = np.nan_to_num(distances_matrix, nan=np.inf)

    if np.any(np.isinf(distances_matrix)):
        print("Infinite values found in the distance matrix. Adjusting them.")
        distances_matrix[distances_matrix == np.inf] = np.max(distances_matrix[np.isfinite(distances_matrix)]) * 1e6

    try:
        row_ind, col_ind = linear_sum_assignment(distances_matrix)
    except ValueError as e:
        print(f"Error during linear sum assignment: {e}")
        return pd.DataFrame()  # Return an empty dataframe if assignment fails
    
    assignments = []
    
    for i, gt_index in enumerate(row_ind):
        gt_study_id = distances_pivot.index[gt_index]
        gen_sampling_method, gen_p, gen_temperature, gen_backward_length = distances_pivot.columns[col_ind[i]]
        group = ground_truth_df.loc[ground_truth_df['study_id'] == gt_study_id, 'group'].iloc[0]
        distance = distances_matrix[gt_index, col_ind[i]]
        
        assignments.append({
            'gt_study_id': gt_study_id,
            'group' : group,
            'gen_sampling_method': gen_sampling_method,
            'gen_p': gen_p,
            'gen_temperature': gen_temperature,
            'gen_backward_length': gen_backward_length,
            'distance': distance
        })
    
    assignments_df = pd.DataFrame(assignments)
    save_dataframe(assignments_df, 'assignmnents')

    
    return assignments_df


# def excel(data):

#     # Save the data to an Excel file
#     data.to_excel("/home/ubuntu/helene/results/assignments.xlsx", index=False)


if __name__ == "__main__":
    ground_truth, generated_text=load_data()
    print(ground_truth.columns)
    ground_truth=drop_most_correlated_metric(ground_truth)
    print(ground_truth.columns)

    distances_df=calculate_distances(ground_truth, generated_text)
    assignments = optimize_assignment(distances_df, ground_truth)
    print(f"Assignments: {assignments}")
    # excel(assignments)
