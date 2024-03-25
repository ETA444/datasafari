from scipy.stats import shapiro, anderson, normaltest
from statsmodels.stats.diagnostic import lilliefors


def evaluate_normality(df, target_variable, grouping_variable, method: str = 'consensus', pipeline: bool = False):
    groups = df[grouping_variable].unique().tolist()

    # define output objects
    output_info = {}  # non-pipeline
    normality_info = {}  # pipeline single-method
    normality_consensus = None  # pipeline consensus method

    if method in ['shapiro', 'consensus']:
        # calculating statistic and p-values to define normality
        shapiro_stats = [shapiro(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        shapiro_pvals = [shapiro(df[df[grouping_variable] == group][target_variable]).pvalue for group in groups]
        shapiro_normality = [True if p > 0.05 else False for p in shapiro_pvals]

        # save the info for return and text for output
        shapiro_info = {group: {'stat': shapiro_stats[n], 'p': shapiro_pvals[n], 'normality': shapiro_normality[n]} for n, group in enumerate(groups)}
        shapiro_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in shapiro_info.items()]
        shapiro_title = f"< NORMALITY TESTING: SHAPIRO-WILK >\n\n"
        shapiro_tip = "☻ Tip: The Shapiro-Wilk test is particularly well-suited for small sample sizes (n < 50) and is considered one of the most powerful tests for assessing normality. It is sensitive to departures from normality, making it a preferred choice for rigorous normality assessments in small datasets.\n"

        # save info in return object and conditionally print to console
        output_info['shapiro'] = shapiro_info
        normality_info['shapiro_group_consensus'] = all(shapiro_normality)

        # end it here if non-consensus method
        if method == 'shapiro':
            print(shapiro_title, *shapiro_text, shapiro_tip)
            return output_info if not pipeline else normality_info['shapiro_group_consensus']

    if method in ['anderson', 'consensus']:
        # calculating statistic to define normality
        anderson_stats = [anderson(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        anderson_critical_values = [anderson(df[df[grouping_variable] == group][target_variable]).critical_values[2] for group in groups]
        anderson_normality = [True if c_val > anderson_stats[n] else False for n, c_val in enumerate(anderson_critical_values)]

        # save the info for return and text for output
        anderson_info = {group: {'stat': anderson_stats[n], 'p': anderson_critical_values[n], 'normality': anderson_normality[n]} for n, group in enumerate(groups)}
        anderson_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in anderson_info.items()]
        anderson_title = f"< NORMALITY TESTING: ANDERSON-DARLING >\n\n"
        anderson_tip = "☻ Tip: The Anderson-Darling test is a versatile test that can be applied to any sample size and is especially useful for comparing against multiple distribution types, not just the normal. It places more emphasis on the tails of the distribution than the Shapiro-Wilk test, making it useful for detecting outliers or heavy-tailed distributions.\n"

        # saving info
        output_info['anderson'] = anderson_info
        normality_info['anderson_group_consensus'] = all(anderson_normality)

        # end it here if non-consensus method
        if method == 'anderson':
            print(anderson_title, *anderson_text, anderson_tip)
            return output_info if not pipeline else normality_info['anderson_group_consensus']

    if method in ['normaltest', 'consensus']:
        # calculating statistic and p-values to define normality
        normaltest_stats = [normaltest(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        normaltest_pvals = [normaltest(df[df[grouping_variable] == group][target_variable]).pvalue for group in groups]
        normaltest_normality = [True if p > 0.05 else False for p in normaltest_pvals]

        # save the info for return and text for output
        normaltest_info = {group: {'stat': normaltest_stats[n], 'p': normaltest_pvals[n], 'normality': normaltest_normality[n]} for n, group in enumerate(groups)}
        normaltest_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in normaltest_info.items()]
        normaltest_title = f"< NORMALITY TESTING: D'AGOSTINO-PEARSON NORMALTEST >\n\n"
        normaltest_tip = "☻ Tip: The D'Agostino-Pearson normality test, or simply 'normaltest', is best applied when the sample size is larger, as it combines skewness and kurtosis to form a test statistic. This test is useful for detecting departures from normality that involve asymmetry and tail thickness, offering a good balance between sensitivity and specificity in medium to large sample sizes.\n"

        # saving info
        output_info['normaltest'] = normaltest_info
        normality_info['normaltest_group_consensus'] = all(normaltest_normality)

        # end it here if non-consensus method
        if method == 'normaltest':
            print(normaltest_title, *normaltest_text, normaltest_tip)
            return output_info if not pipeline else normality_info['normaltest_group_consensus']

    if method in ['lilliefors', 'consensus']:
        # calculating statistic and p-values to define normality
        lilliefors_stats = [lilliefors(df[df[grouping_variable] == group][target_variable])[0] for group in groups]
        lilliefors_pvals = [lilliefors(df[df[grouping_variable] == group][target_variable])[1] for group in groups]
        lilliefors_normality = [True if p > 0.05 else False for p in lilliefors_pvals]

        # save the info for return and text for output
        lilliefors_info = {group: {'stat': lilliefors_stats[n], 'p': lilliefors_pvals[n], 'normality': lilliefors_normality[n]} for n, group in enumerate(groups)}
        lilliefors_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in lilliefors_info.items()]
        lilliefors_title = f"< NORMALITY TESTING: LILLIEFORS' TEST >\n\n"
        lilliefors_tip = "☻ Tip: The Lilliefors test is an adaptation of the Kolmogorov-Smirnov test for normality with the benefit of not requiring the mean and variance to be known parameters. It's particularly useful for small to moderately sized samples and is sensitive to deviations from normality in the center of the distribution rather than the tails. This makes it complementary to tests like the Anderson-Darling when a comprehensive assessment of normality is needed.\n"

        # saving info
        output_info['lilliefors'] = lilliefors_info
        normality_info['lilliefors_group_consensus'] = all(lilliefors_normality)

        # end here if non-consensus method
        if method == 'lilliefors':
            print(lilliefors_title, *lilliefors_text, lilliefors_tip)
            return output_info if not pipeline else normality_info['lilliefors_group_consensus']

    if method == 'consensus':
        # the logic is that more than half of the tests need to give True for the consensus to be True
        true_count = [group_normality_consensus for test_name, group_normality_consensus in normality_info.items()].count(True)
        false_count = [group_normality_consensus for test_name, group_normality_consensus in normality_info.items()].count(False)
        half_point = len(normality_info)/2
        if true_count > half_point:
            consensus_percent = (true_count / len(normality_info)) * 100
            normality_consensus_result = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest Normality. *\n\n* Detailed results of each test are provided below.\n"

            # used only within predict_hypothesis() pipeline
            normality_consensus = True

        elif true_count < half_point:
            consensus_percent = (false_count / len(normality_info)) * 100
            normality_consensus_result = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest Non-Normality. *\n\n* Detailed results of each test are provided below:\n"

            # used only within predict_hypothesis() pipeline
            normality_consensus = False

        elif true_count == half_point:
            # the logic is that we default to the decisions of shapiro-wilk and anderson-darling (if both true then true) - internal use only
            normality_consensus_result = f"  ➡ Result: Consensus is not reached. (50% Normality / 50% Non-normality)\n\n∴ Please refer to the results of each test below:\n"

            # used only within predict_hypothesis() pipeline
            normality_consensus = all([normality_info['shapiro_group_consensus'], normality_info['anderson_group_consensus']])

        # construct output and return
        print(f"< NORMALITY TESTING: CONSENSUS >\nThe consensus method bases its conclusion on 4 tests: Shapiro-Wilk test, Anderson-Darling test, D'Agostino-Pearson test, Lilliefors test. (Note: More than 50% must have the same outcome to reach consensus. In predict_hypothesis(), if consensus is not reached normality is settled based on Shapiro-Wilk and Anderson-Darling outcomes)\n\n{normality_consensus_result}")
        print(shapiro_title, *shapiro_text, shapiro_tip)
        print(anderson_title, *anderson_text, anderson_tip)
        print(normaltest_title, *normaltest_text, normaltest_tip)
        print(lilliefors_title, *lilliefors_text, lilliefors_tip)
        return output_info if not pipeline else normality_consensus
