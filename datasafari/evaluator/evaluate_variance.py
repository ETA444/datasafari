from scipy.stats import levene, bartlett, fligner


def evaluate_variance(df, target_variable, grouping_variable, normality_info: bool = None, method: str = 'consensus', pipeline: bool = False):
    groups = df[grouping_variable].unique().tolist()
    samples = [df[df[grouping_variable] == group][target_variable] for group in groups]

    # define output objects
    output_info = {}  # non-pipeline
    variance_info = {}  # pipeline single-method
    variance_consensus = None  # pipeline consensus method

    if method in ['levene', 'consensus']:
        levene_stat, levene_pval = levene(*samples).statistic, levene(*samples).pvalue
        variance_info['levene'] = levene_pval > 0.05

        # save the info in return object
        output_info['levene'] = {'stat': levene_stat, 'p': levene_pval, 'equal_variances': variance_info['levene']}

        # construct console output
        levene_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {levene_stat}\n  ➡ p-value: {levene_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['levene'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        levene_title = f"< EQUAL VARIANCES TESTING: LEVENE >\n\n"
        levene_tip = "☻ Tip: The Levene test is robust against departures from normality, making it suitable for a wide range of distributions, including those that are not normally distributed. It is preferred when you suspect that your data may not adhere to the normality assumption.\n"

        # output & return
        if method == 'levene':
            print(levene_title, levene_text, levene_tip)
            return output_info if not pipeline else variance_info['levene']

    if method in ['fligner', 'consensus']:
        fligner_stat, fligner_pval = fligner(*samples).statistic, fligner(*samples).pvalue
        variance_info['fligner'] = fligner_pval > 0.05

        # save the info in return object
        output_info['fligner'] = {'stat': fligner_stat, 'p': fligner_pval, 'equal_variances': variance_info['fligner']}

        # construct console output
        fligner_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {fligner_stat}\n  ➡ p-value: {fligner_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['fligner'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        fligner_title = f"< EQUAL VARIANCES TESTING: FLIGNER-KILLEEN >\n\n"
        fligner_tip = "☻ Tip: The Fligner-Killeen test is a non-parametric alternative that is less sensitive to departures from normality compared to the Bartlett test. It's a good choice when dealing with data that might not be normally distributed, offering robustness similar to the Levene test but through a different statistical approach.\n"

        # output & return
        if method == 'fligner':
            print(fligner_title, fligner_text, fligner_tip)
            return output_info if not pipeline else variance_info['fligner']

    if method in ['bartlett', 'consensus'] and normality_info:
        bartlett_stat, bartlett_pval = bartlett(*samples).statistic, bartlett(*samples).pvalue
        variance_info['bartlett'] = bartlett_pval > 0.05

        # save the info in return object
        output_info['bartlett'] = {'stat': bartlett_stat, 'p': bartlett_pval, 'equal_variances': variance_info['bartlett']}

        # construct console output
        bartlett_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {bartlett_stat}\n  ➡ p-value: {bartlett_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['bartlett'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        bartlett_title = f"< EQUAL VARIANCES TESTING: BARTLETT >\n\n"
        bartlett_tip = "☻ Tip: The Fligner-Killeen test is a non-parametric alternative that is less sensitive to departures from normality compared to the Bartlett test. It's a good choice when dealing with data that might not be normally distributed, offering robustness similar to the Levene test but through a different statistical approach.\n"

        # output & return
        if method == 'bartlett':
            print(bartlett_title, bartlett_text, bartlett_tip)
            return output_info if not pipeline else variance_info['bartlett']

    if method == 'consensus':
        # the logic is that more than half of the tests need to give True for the consensus to be True
        variance_results = [variance_bool for variance_bool in variance_info.values()]
        true_count = variance_results.count(True)
        false_count = variance_results.count(False)
        half_point = 1.5 if len(variance_results) == 3 else 1
        if true_count > half_point:
            consensus_percent = (true_count / len(variance_results)) * 100
            variance_consensus_text = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest equal variance between samples. *\n\n* Detailed results of each test are provided below.\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = True

        elif true_count < half_point:
            consensus_percent = (false_count / len(variance_results)) * 100
            variance_consensus_text = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest unequal variance between samples. *\n\n* Detailed results of each test are provided below:\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = False

        elif true_count == half_point:
            # the logic is that we default to the decisions of levene - internal use only
            variance_consensus_text = f"  ➡ Result: Consensus is not reached.\n\n∴ Please refer to the results of each test below:\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = variance_info['levene']

        print(f"< VARIANCE TESTING: CONSENSUS >\nThe consensus method bases its conclusion on 2-3 tests: Levene test, Fligner-Killeen test, Bartlett test. (Note: More than 50% must have the same outcome to reach consensus.)\n\n{variance_consensus_text}")
        print(levene_title, levene_text, levene_tip)
        print(fligner_title, fligner_text, fligner_tip)
        print(bartlett_title, bartlett_text, bartlett_tip) if normality_info else f"\n\n< NOTE ON BARTLETT >\nBartlett was not used in consensus as no normality info has been provided or data is non-normal. Accuracy of Bartlett's test results rely heavily on normality."

        return output_info if not pipeline else variance_consensus
