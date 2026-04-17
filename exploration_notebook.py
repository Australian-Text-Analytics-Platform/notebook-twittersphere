# class to drive exploration notebook

import os
import pandas as pd
import numpy as np
from typing import Optional, List
import plotly.express as px
import plotly.io as pio
import requests
from tqdm import tqdm


def create_pattern_match(keyword, whole_word=True):
    """
    Utility function to create search pattern for keyword regardless of the column.

    ngrams are encoded as "{word} {word} {word}" or "{word}" - so we need to match on
    either the separating space, or the start/end of the string.

    """

    if whole_word:
        # ngrams are stored separated by spaces
        pattern = f'(?:^|(?<= )){keyword}(?:$|(?= ))'
    else:
        pattern = keyword

    return pattern


class Exploration:
    def __init__(self) -> None:
        pass

    
    def get_data(self, url, filename, force = False) -> str:
        cwd = os.getcwd()
        folder = os.path.join(cwd, "data")
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath) and not force:
            return filepath
        
        response = requests.get(url, stream=True)
        
        # Sizes in bytes.
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")

        return filepath

    def top_grams_in_date_range(self, data_loc: str, start_date: str, end_date: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Filter data for top N one-grams within a specified date range,
        based on their summed frequencies.

        Args:
            data (str): DataFrame with 'date', 'gram', and 'frequency' columns.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            top_n (int): Number of top grams to return. Defaults to 20.

        Returns:
            pd.DataFrame: DataFrame with 'gram' and 'frequency' of top N grams,
                          or None if an error occurs.
        """
        try:

            filtered_data = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date)))

            # Group by 'ngram' and sum 'total_frequency'
            # Assuming 'ngram' column for words/n-grams and 'total_frequency' for their counts
            gram_frequencies = filtered_data.groupby('ngram')['total_frequency'].sum()

            # Sort by frequency in descending order and get top N
            top_grams_series = gram_frequencies.sort_values(ascending=False).head(top_n)

            # Convert the resulting Series to a DataFrame
            top_grams_df = top_grams_series.reset_index()
            
            return top_grams_df
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'gram' and 'frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error filtering and ranking data: {e}")
            return None

    def keyword_search_in_date_range(self, data_loc: str, keyword: str, start_date: str, end_date: str, whole_word=True) -> Optional[pd.DataFrame]:
        """
        Search for a keyword (e.g., 1-gram) within n-grams in a specified date range
        and return its daily occurrences and frequencies. Matches keyword as a whole word, case-insensitively.

        Args:
            data_loc (str): DataFrame with 'date', 'ngram', and 'total_frequency' columns.
            keyword (str): The word to search for within the 'ngram' column.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with 'date' and 'total_frequency' for n-grams containing the keyword,
                          or None if an error occurs or keyword not found.
        """

        # Find ngrams used in the range
        filtered_data = pd.read_parquet(data_loc, columns=['ngram'], filters=(('date', '>=', start_date), ('date', '<=', end_date)))

        # Just grab the unique ngrams
        unique_ngrams = pd.Series(pd.unique(filtered_data['ngram']))

        search_pattern = create_pattern_match(keyword, whole_word)

        keyword_mask = unique_ngrams.str.contains(search_pattern, case=False, na=False, regex=True)
        matching_ngrams = unique_ngrams.loc[keyword_mask]

        # Select relevant columns. Include 'ngram' to show which n-gram matched.
        return pd.read_parquet(data_loc, columns=['date', 'ngram', 'total_frequency'], filters=(('date', '>=', start_date), ('date', '<=', end_date), ('ngram', 'in', matching_ngrams)))

            
    def keyword_search_with_ratios_in_date_range(self, data_loc: str, keyword: str, start_date: str, end_date: str, whole_word=True) -> Optional[pd.DataFrame]:
        """
        Search for a keyword within n-grams in a date range and return aggregated total frequency,
        component frequencies (retweet, quote, reply, original), and their ratios
        to the total frequency. Matches keyword as a whole word, case-insensitively.

        Args:
            data_loc (str): parquet file with 'date', 'ngram', 'total_frequency',
                                 'retweet_frequency', 'quote_tweet_frequency',
                                 'reply_tweet_frequency', 'original_tweet_frequency' columns.
            keyword (str): The word to search for within the 'ngram' column (e.g., 1-gram or part of a 3-gram).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            whole_word (bool): whether to match whole words or match anywhere.

        Returns:
            pd.DataFrame: A DataFrame with one row containing the keyword, summed frequencies for matching n-grams,
                          and ratios. Returns an empty DataFrame if keyword not found or
                          relevant frequency columns are missing. Returns None on other errors.
        """

        # Find ngrams used in the range
        filtered_data = pd.read_parquet(data_loc, columns=['ngram'], filters=(('date', '>=', start_date), ('date', '<=', end_date)))

        # Just grab the unique ngrams
        unique_ngrams = pd.Series(pd.unique(filtered_data['ngram']))

        search_pattern = create_pattern_match(keyword, whole_word)

        keyword_mask = unique_ngrams.str.contains(search_pattern, case=False, na=False, regex=True)
        matching_ngrams = unique_ngrams.loc[keyword_mask]

        # Matching columns for further processing
        ngram_frequencies = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date), ('ngram', 'in', matching_ngrams)))

        # Sum frequencies for the n-grams containing the keyword in the date range
        frequency_columns = [
            'total_frequency', 'retweet_frequency', 'quote_tweet_frequency',
            'reply_tweet_frequency', 'original_tweet_frequency'
        ]
        summed_frequencies = ngram_frequencies[frequency_columns].sum()

        total_freq_sum = summed_frequencies['total_frequency']

        results = {
            'keyword': keyword,
            'total_frequency_sum': total_freq_sum
        }

        ratio_results = {}

        # Calculate sums and ratios for component frequencies
        for col_name in frequency_columns:
            sum_col_name = f"{col_name}_sum"
            ratio_col_name = f"{col_name.replace('_frequency', '')}_ratio"
            
            current_sum = summed_frequencies[col_name]
            results[sum_col_name] = current_sum
            
            if total_freq_sum > 0:
                ratio_results[ratio_col_name] = current_sum / total_freq_sum
            else:
                ratio_results[ratio_col_name] = 0.0 # Or np.nan, depending on desired output for 0/0

        results.update(ratio_results)
        
        return pd.DataFrame([results])
    
    
    def plot_keyword_frequencies_comparison(self, data_loc: pd.DataFrame, keywords_list: List[str], start_date: str, end_date: str) -> None:
        """
        Plots a line graph comparing the daily frequencies of a list of keywords (up to 10)
        within a specified time range. Search is case-insensitive and matches whole words.

        Args:
            data_loc (str): path to the parquet file to query.
            keywords_list (List[str]): A list of keywords to compare (max 10).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        if not keywords_list:
            print("Error: No keywords provided for plotting.")
            return
        if len(keywords_list) > 30:
            print("Error: Maximum of 30 keywords allowed for comparison.")
            return

        # Assemble keywords dataframe combining all of them
        keywords_results = pd.concat(
            self.keyword_search_in_date_range(data_loc, keyword, start_date, end_date)
            for keyword in keywords_list
        )

        fig = px.line(keywords_results, x='date', y='total_frequency', color='ngram',
                      title=f'Keyword Frequency Comparison ({start_date} to {end_date})',
                      labels={'date': 'Date', 'frequency': 'Total Daily Frequency'})

        return fig


    def plot_top_n_grams_trend(self, data_loc: str, start_date: str, end_date: str, top_n: int = 10) -> None:
        """
        Identifies the top N n-grams within a specified date range based on their total summed frequency,
        and then plots their daily frequency trends over that period.

        Args:
            data_loc (str): path to the parquet file to query.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            top_n (int): Number of top n-grams to identify and plot. Defaults to 10.
        """

        if top_n <= 0:
            print("Error: top_n must be a positive integer.")
            return
        if top_n > 20: # Limiting to 20 for plot readability, can be adjusted
            print("Warning: Plotting more than 20 n-grams might make the graph cluttered. Consider a smaller top_n.")

        top_ngrams = self.top_grams_in_date_range(data_loc, start_date, end_date)

        # Assemble keywords dataframe combining all of them
        keywords_results = pd.concat(
            self.keyword_search_in_date_range(data_loc, keyword, start_date, end_date)
            for keyword in top_ngrams['ngram']
        )

        fig = px.line(keywords_results, x='date', y='total_frequency', color='ngram',
                      title=f'Keyword Frequency Comparison ({start_date} to {end_date})',
                      labels={'date': 'Date', 'frequency': 'Total Daily Frequency'})

        return fig

    # --- Hashtag Specific Methods ---

    def top_hashtags_in_date_range(self, data_loc: str, start_date: str, end_date: str, top_n: int = 20, normalise_case=True) -> Optional[pd.DataFrame]:
        """
        Filter data for top N hashtags within a specified date range,
        based on their summed frequencies. Assumes 'hashtag' and 'total_frequency' columns.
        
        """

        filtered_data = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date)))

        if normalise_case:
            filtered_data['hashtag'] = filtered_data['hashtag'].str.lower()

        hashtag_frequencies = filtered_data.groupby('hashtag')['total_frequency'].sum()
        top_hashtags_series = hashtag_frequencies.sort_values(ascending=False).head(top_n)
        
        return top_hashtags_series.reset_index()

    def search_hashtag_in_date_range(self, data_loc: str, hashtag_to_search: str, start_date: str, end_date: str, match_whole_hashtag=True, normalise_case=True) -> Optional[pd.DataFrame]:
        """
        Search for a hashtag within a specified date range and return its daily occurrences and frequencies.
        Matches hashtag as a whole word, case-insensitively by default, otherwise performs a regex.

        Assumes 'hashtag' and 'total_frequency' columns.

        """


        filtered_data = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date)))
        
        search_pattern = hashtag_to_search
        
        if match_whole_hashtag:
            search_pattern = f'^{hashtag_to_search}$'

        matches = filtered_data['hashtag'].str.contains(search_pattern, case=False, na=False, regex=True)

        filtered_data = filtered_data[matches]

        if normalise_case:
            filtered_data['hashtag'] = filtered_data['hashtag'].str.lower()
            filtered_data = filtered_data.groupby(['hashtag', 'date']).sum()
            filtered_data = filtered_data.reset_index()

        return filtered_data[['hashtag', 'date', 'total_frequency']]

    def search_hashtag_total_frequency_in_range(self, data_loc: str, hashtag_keyword: str, start_date: str, end_date: str, match_whole_hashtag=True, normalise_case=True) -> Optional[pd.DataFrame]:
        """
        Search for a hashtag within a date range and return its total summed frequency.

        Matches hashtag as a whole word, case-insensitively. Assumes 'hashtag' and 'total_frequency' columns.
        """

        matches = self.search_hashtag_in_date_range(data_loc, hashtag_keyword, start_date, end_date, match_whole_hashtag=match_whole_hashtag)

        total_freq_sum = matches[['hashtag', 'total_frequency']].groupby('hashtag').sum()

        return total_freq_sum

    def plot_hashtag_frequencies_comparison(self, data_loc: str, hashtags_list: List[str], start_date: str, end_date: str) -> None:
        """
        Plots a line graph comparing the daily frequencies of a list of hashtags (up to 10)
        within a specified time range. Assumes 'hashtag' and 'total_frequency' columns.

        """
        if not hashtags_list:
            print("Error: No hashtags provided for plotting.")
            return
        if len(hashtags_list) > 10:
            print("Error: Maximum of 10 hashtags allowed for comparison.")
            return

        # Assemble hashtags dataframe combining all of them
        keywords_results = pd.concat(
            self.search_hashtag_in_date_range(data_loc, hashtag, start_date, end_date)
            for hashtag in hashtags_list
        )

        fig = px.line(keywords_results, x='date', y='total_frequency', color='hashtag',
                      title=f'Keyword Frequency Comparison ({start_date} to {end_date})',
                      labels={'date': 'Date', 'frequency': 'Total Daily Frequency'})

        return fig

    def plot_top_hashtags_trend(self, data_loc: str, start_date: str, end_date: str, top_n: int = 10) -> None:
        """
        Identifies the top N hashtags within a specified date range based on their total summed frequency,
        and then plots their daily frequency trends over that period. Assumes 'hashtag' and 'total_frequency' columns.
        """
        if top_n <= 0:
            print("Error: top_n must be a positive integer.")
            return
        if top_n > 20:
            print("Warning: Plotting more than 20 hashtags might make the graph cluttered.")

        top_hashtags = self.top_hashtags_in_date_range(data_loc, start_date, end_date, top_n=top_n)

        return self.plot_hashtag_frequencies_comparison(data_loc, list(top_hashtags['hashtag']), start_date, end_date)

    # --- Domain Specific Methods ---

    def top_domains_in_date_range(self, data_loc: str, start_date: str, end_date: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Filter data for top N domains within a specified date range,
        based on their summed frequencies. Assumes 'domain' and 'total_frequency' columns.
        """
        try:

            filtered_data = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date)))

            # Group by 'domain' and sum 'total_frequency'
            # Assuming 'domain' column for domains and 'total_frequency' for their counts
            domain_frequencies = filtered_data.groupby('domain')['total_frequency'].sum()

            # Sort by frequency in descending order and get top N
            top_domains_series = domain_frequencies.sort_values(ascending=False).head(top_n)

            # Convert the resulting Series to a DataFrame
            top_domains_df = top_domains_series.reset_index()
            
            return top_domains_df
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'domain' and 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error filtering and ranking domains: {e}")
            return None

    def search_domain_in_date_range(self, data_loc: str, domain_to_search: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a domain keyword within a specified date range and return its daily occurrences and frequencies.
        Matches keyword as whole 'domain' column. Assumes 'domain' and 'total_frequency' columns.
        """
        try:
            filtered_data = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date), ('domain', '=', domain_to_search)))

            # Convert the resulting Series to a DataFrame
            # Discard specific retweet_frequency, quote_tweet_frequency, reply_tweet_frequency, original_tweet_frequency columns
            domain_search_df = filtered_data[['date', 'domain', 'total_frequency']].sort_values(by='date').reset_index()

            return domain_search_df
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'domain', 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error during domain search: {e}")
            return None

    def search_domain_total_frequency_in_range(self, data_loc: str, domain_keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a domain keyword within a date range and return its total summed frequency.
        Matches keyword as whole 'domain' column. Assumes 'domain' and 'total_frequency' columns.
        """

        filtered_data = self.search_domain_in_date_range(data_loc, domain_keyword, start_date, end_date)

        total_freq_sum = filtered_data['total_frequency'].sum()
        results = {'domain': domain_keyword, 'total_frequency_sum': total_freq_sum}
        return pd.DataFrame([results])

    def plot_domain_frequencies_comparison(self, data_loc: str, domains_list: List[str], start_date: str, end_date: str) -> None:
        """
        Plots a line graph comparing the daily frequencies of a list of domain keywords (up to 10)
        within a specified time range. Assumes 'domain' and 'total_frequency' columns.
        """
        if not domains_list:
            print("Error: No domains provided for plotting.")
            return
        if len(domains_list) > 10:
            print("Error: Maximum of 10 domains allowed for comparison.")
            return

        try:

            relevant_data = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date), ('domain', 'in', domains_list)))

            # date_mask_overall = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            # relevant_data = data_copy[date_mask_overall]

            if relevant_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            all_domains_df = pd.DataFrame()

            for dom in domains_list:
                keyword_mask = relevant_data['domain'] == dom
                domain_data = relevant_data[keyword_mask]

                if not domain_data.empty:
                    daily_freq = domain_data.groupby('date')['total_frequency'].sum().reset_index()
                    daily_freq = daily_freq.rename(columns={'total_frequency': dom})
                    
                    if all_domains_df.empty:
                        all_domains_df = daily_freq
                    else:
                        all_domains_df = pd.merge(all_domains_df, daily_freq, on='date', how='outer')
                else:
                    print(f"Domain containing '{dom}' not found in the specified date range.")
            
            if all_domains_df.empty:
                print("No data found for any of the specified domains in the date range.")
                return

            all_domains_df = all_domains_df.set_index('date').fillna(0).reset_index()
            plot_df = all_domains_df.melt(id_vars=['date'], value_vars=domains_list,
                                          var_name='domain', value_name='frequency')

            if plot_df.empty:
                print("No frequency data to plot after processing domains.")
                return

            fig = px.line(plot_df, x='date', y='frequency', color='domain',
                          title=f'Domain Keyword Frequency Comparison ({start_date} to {end_date})',
                          labels={'date': 'Date', 'frequency': 'Total Daily Frequency', 'domain': 'Domain Keyword'})
            fig.show()
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
        except Exception as e:
            print(f"Error generating domain comparison plot: {e}")

    def plot_top_domains_trend(self, data_loc: str, start_date: str, end_date: str, top_n: int = 10) -> None:
        """
        Identifies the top N domains within a specified date range based on their total summed frequency,
        and then plots their daily frequency trends over that period. Assumes 'domain' and 'total_frequency' columns.
        """
        if top_n <= 0:
            print("Error: top_n must be a positive integer.")
            return
        if top_n > 20:
            print("Warning: Plotting more than 20 domains might make the graph cluttered.")

        try:
            period_data = pd.read_parquet(data_loc, filters=(('date', '>=', start_date), ('date', '<=', end_date)))

            if period_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            top_domains_overall = period_data.groupby('domain')['total_frequency'].sum().nlargest(top_n).index.tolist()

            if not top_domains_overall:
                print(f"Could not determine top {top_n} domains for the period.")
                return

            all_top_domains_df = pd.DataFrame()

            for dom_to_plot in top_domains_overall:
                domain_mask = period_data['domain'] == dom_to_plot # Exact match for already identified top domains
                daily_data_for_domain = period_data[domain_mask]

                if not daily_data_for_domain.empty:
                    daily_freq = daily_data_for_domain.groupby('date')['total_frequency'].sum().reset_index()
                    daily_freq = daily_freq.rename(columns={'total_frequency': dom_to_plot})
                    
                    if all_top_domains_df.empty:
                        all_top_domains_df = daily_freq
                    else:
                        all_top_domains_df = pd.merge(all_top_domains_df, daily_freq, on='date', how='outer')
            
            if all_top_domains_df.empty:
                print("No daily frequency data found for any of the top domains.")
                return
            
            all_top_domains_df = all_top_domains_df.set_index('date').fillna(0).reset_index()
            plot_df = all_top_domains_df.melt(id_vars=['date'], value_vars=top_domains_overall,
                                              var_name='domain', value_name='frequency')

            if plot_df.empty:
                print("No frequency data to plot after processing top domains.")
                return

            fig = px.line(plot_df, x='date', y='frequency', color='domain',
                          title=f'Daily Frequency Trend of Top {top_n} Domains ({start_date} to {end_date})',
                          labels={'date': 'Date', 'frequency': 'Total Daily Frequency', 'domain': 'Domain'})
            fig.show()
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
        except Exception as e:
            print(f"Error generating top domains trend plot: {e}")