# class to drive exploration notebook

import os
import pandas as pd
import numpy as np
from typing import Optional, List
import plotly.express as px
import plotly.io as pio
import requests
from tqdm import tqdm

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

    def load_data(self, path: str) -> Optional[pd.DataFrame]:
        """
        Load data from a CSV file.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            data = pd.read_csv(path)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def top_grams_in_date_range(self, data: pd.DataFrame, start_date: str, end_date: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Filter data for top N one-grams within a specified date range,
        based on their summed frequencies.

        Args:
            data (pd.DataFrame): DataFrame with 'date', 'gram', and 'frequency' columns.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            top_n (int): Number of top grams to return. Defaults to 20.

        Returns:
            pd.DataFrame: DataFrame with 'gram' and 'frequency' of top N grams,
                          or None if an error occurs.
        """
        try:
            # Ensure 'date' column is in datetime format
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            # Filter data by date range
            mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            filtered_data = data_copy.loc[mask]

            if filtered_data.empty:
                return pd.DataFrame(columns=['ngram', 'total_frequency'])

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

    def keyword_search_in_date_range(self, data: pd.DataFrame, keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a keyword (e.g., 1-gram) within n-grams in a specified date range
        and return its daily occurrences and frequencies. Matches keyword as a whole word, case-insensitively.

        Args:
            data (pd.DataFrame): DataFrame with 'date', 'ngram', and 'total_frequency' columns.
            keyword (str): The word to search for within the 'ngram' column.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with 'date' and 'total_frequency' for n-grams containing the keyword,
                          or None if an error occurs or keyword not found.
        """
        try:
            data_copy = data.copy()
            # Ensure 'date' column is in datetime format
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            # Filter data by date range
            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            
            # Filter by keyword (whole word, case-insensitive)
            # Ensure 'ngram' is string type to use .str accessor
            if not pd.api.types.is_string_dtype(data_copy['ngram']):
                data_copy['ngram'] = data_copy['ngram'].astype(str)
            keyword_mask = data_copy['ngram'].str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
            
            # Combine masks
            combined_mask = date_mask & keyword_mask
            
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Keyword '{keyword}' not found as a whole word in any n-gram within the specified date range.")
                return pd.DataFrame(columns=['date', 'ngram', 'total_frequency'])

            # Select relevant columns. Include 'ngram' to show which n-gram matched.
            result_df = filtered_data[['date', 'ngram', 'total_frequency']].sort_values(by='date')
            
            return result_df
            
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'ngram', and 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error during keyword search: {e}")
            return None
        
    def emoji_search_in_date_range(self, data: pd.DataFrame, keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a keyword (e.g., emoji) within n-grams in a specified date range
        and return its daily occurrences and frequencies. Matches keyword as a whole word, case-insensitively.

        Args:
            data (pd.DataFrame): DataFrame with 'date', 'ngram', and 'total_frequency' columns.
            keyword (str): The word to search for within the 'ngram' column.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with 'date' and 'total_frequency' for n-grams containing the keyword,
                          or None if an error occurs or keyword not found.
        """
        try:
            data_copy = data.copy()
            # Ensure 'date' column is in datetime format
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            # Filter data by date range
            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            
            # Filter by keyword (whole word, case-insensitive)
            # Ensure 'ngram' is string type to use .str accessor
            if not pd.api.types.is_string_dtype(data_copy['ngram']):
                data_copy['ngram'] = data_copy['ngram'].astype(str)
            keyword_mask = data_copy['ngram'].astype(str).str.contains(keyword, case=False, na=False, regex=False)
            
            # Combine masks
            combined_mask = date_mask & keyword_mask
            
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Keyword '{keyword}' not found as a whole word in any n-gram within the specified date range.")
                return pd.DataFrame(columns=['date', 'ngram', 'total_frequency'])

            # Select relevant columns. Include 'ngram' to show which n-gram matched.
            result_df = filtered_data[['date', 'ngram', 'total_frequency']].sort_values(by='date')
            
            return result_df
            
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'ngram', and 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error during keyword search: {e}")
            return None

    def keyword_search_with_ratios_in_date_range(self, data: pd.DataFrame, keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a keyword within n-grams in a date range and return aggregated total frequency,
        component frequencies (retweet, quote, reply, original), and their ratios
        to the total frequency. Matches keyword as a whole word, case-insensitively.

        Args:
            data (pd.DataFrame): DataFrame with 'date', 'ngram', 'total_frequency',
                                 'retweet_frequency', 'quote_tweet_frequency',
                                 'reply_tweet_frequency', 'original_tweet_frequency' columns.
            keyword (str): The word to search for within the 'ngram' column (e.g., 1-gram or part of a 3-gram).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame with one row containing the keyword, summed frequencies for matching n-grams,
                          and ratios. Returns an empty DataFrame if keyword not found or
                          relevant frequency columns are missing. Returns None on other errors.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            # Define frequency columns
            frequency_columns = [
                'total_frequency', 'retweet_frequency', 'quote_tweet_frequency',
                'reply_tweet_frequency', 'original_tweet_frequency'
            ]
            
            # Check if all necessary columns exist
            required_columns = ['date', 'ngram'] + frequency_columns
            for col in required_columns:
                if col not in data_copy.columns:
                    print(f"Error: Missing required column '{col}'.")
                    return pd.DataFrame()

            # Ensure 'ngram' is string type to use .str accessor
            if not pd.api.types.is_string_dtype(data_copy['ngram']):
                data_copy['ngram'] = data_copy['ngram'].astype(str)

            # Filter by date range and keyword (whole word, case-insensitive)
            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            keyword_mask = data_copy['ngram'].str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
            combined_mask = date_mask & keyword_mask
            
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Keyword '{keyword}' not found as a whole word in any n-gram within the specified date range.")
                return pd.DataFrame()

            # Sum frequencies for the n-grams containing the keyword in the date range
            summed_frequencies = filtered_data[frequency_columns].sum()

            total_freq_sum = summed_frequencies['total_frequency']

            results = {
                'keyword': keyword,
                'total_frequency_sum': total_freq_sum
            }

            ratio_results = {}

            # Calculate sums and ratios for component frequencies
            for col_name in ['retweet_frequency', 'quote_tweet_frequency', 'reply_tweet_frequency', 'original_tweet_frequency']:
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

        except KeyError as e:
            # This specific KeyError for column access should be caught by the check above,
            # but kept as a fallback.
            print(f"Error processing data: Missing column {e}.")
            return None
        except Exception as e:
            print(f"Error during keyword search with ratios: {e}")
            return None
        
    def emoji_search_with_ratios_in_date_range(self, data: pd.DataFrame, keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a keyword/emoji within n-grams in a date range and return aggregated total frequency,
        component frequencies (retweet, quote, reply, original), and their ratios
        to the total frequency. Matches keyword as a whole word, case-insensitively.

        Args:
            data (pd.DataFrame): DataFrame with 'date', 'ngram', 'total_frequency',
                                 'retweet_frequency', 'quote_tweet_frequency',
                                 'reply_tweet_frequency', 'original_tweet_frequency' columns.
            keyword (str): The word to search for within the 'ngram' column (e.g., 1-gram or part of a 3-gram).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame with one row containing the keyword, summed frequencies for matching n-grams,
                          and ratios. Returns an empty DataFrame if keyword not found or
                          relevant frequency columns are missing. Returns None on other errors.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            # Define frequency columns
            frequency_columns = [
                'total_frequency', 'retweet_frequency', 'quote_tweet_frequency',
                'reply_tweet_frequency', 'original_tweet_frequency'
            ]
            
            # Check if all necessary columns exist
            required_columns = ['date', 'ngram'] + frequency_columns
            for col in required_columns:
                if col not in data_copy.columns:
                    print(f"Error: Missing required column '{col}'.")
                    return pd.DataFrame()

            # Ensure 'ngram' is string type to use .str accessor
            if not pd.api.types.is_string_dtype(data_copy['ngram']):
                data_copy['ngram'] = data_copy['ngram'].astype(str)

            # Filter by date range and keyword (whole word, case-insensitive)
            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            keyword_mask = data_copy['ngram'].astype(str).str.contains(keyword, case=False, na=False, regex=False)
            combined_mask = date_mask & keyword_mask
            
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Keyword '{keyword}' not found as a whole word in any n-gram within the specified date range.")
                return pd.DataFrame()

            # Sum frequencies for the n-grams containing the keyword in the date range
            summed_frequencies = filtered_data[frequency_columns].sum()

            total_freq_sum = summed_frequencies['total_frequency']

            results = {
                'keyword': keyword,
                'total_frequency_sum': total_freq_sum
            }

            ratio_results = {}

            # Calculate sums and ratios for component frequencies
            for col_name in ['retweet_frequency', 'quote_tweet_frequency', 'reply_tweet_frequency', 'original_tweet_frequency']:
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

        except KeyError as e:
            # This specific KeyError for column access should be caught by the check above,
            # but kept as a fallback.
            print(f"Error processing data: Missing column {e}.")
            return None
        except Exception as e:
            print(f"Error during keyword search with ratios: {e}")
            return None
    
    
    def plot_keyword_frequencies_comparison(self, data: pd.DataFrame, keywords_list: List[str], start_date: str, end_date: str) -> None:
        """
        Plots a line graph comparing the daily frequencies of a list of keywords (up to 10)
        within a specified time range. Search is case-insensitive and matches whole words.

        Args:
            data (pd.DataFrame): DataFrame with 'date', 'ngram', and 'total_frequency' columns.
            keywords_list (List[str]): A list of keywords to compare (max 10).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        if not keywords_list:
            print("Error: No keywords provided for plotting.")
            return
        if len(keywords_list) > 10:
            print("Error: Maximum of 10 keywords allowed for comparison.")
            return

        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            
            # Ensure 'ngram' is string type for .str accessor
            if not pd.api.types.is_string_dtype(data_copy['ngram']):
                data_copy['ngram'] = data_copy['ngram'].astype(str)

            # Filter data by the overall date range first for efficiency
            date_mask_overall = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            relevant_data = data_copy[date_mask_overall]

            if relevant_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            all_keywords_df = pd.DataFrame()

            for keyword in keywords_list:
                # Filter by keyword (whole word, case-insensitive)
                keyword_mask = relevant_data['ngram'].str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
                keyword_data = relevant_data[keyword_mask]

                if not keyword_data.empty:
                    # Group by date and sum total_frequency
                    daily_freq = keyword_data.groupby('date')['total_frequency'].sum().reset_index()
                    daily_freq = daily_freq.rename(columns={'total_frequency': keyword})
                    
                    if all_keywords_df.empty:
                        all_keywords_df = daily_freq
                    else:
                        all_keywords_df = pd.merge(all_keywords_df, daily_freq, on='date', how='outer')
                else:
                    print(f"Keyword '{keyword}' not found in the specified date range.")
            
            if all_keywords_df.empty:
                print("No data found for any of the specified keywords in the date range.")
                return

            # Fill NaN values that may result from outer merge if keywords don't appear on all same dates
            all_keywords_df = all_keywords_df.set_index('date').fillna(0).reset_index()
            
            # Melt dataframe for Plotly Express
            plot_df = all_keywords_df.melt(id_vars=['date'], value_vars=keywords_list,
                                           var_name='keyword', value_name='frequency')

            if plot_df.empty:
                print("No frequency data to plot after processing keywords.")
                return

            fig = px.line(plot_df, x='date', y='frequency', color='keyword',
                          title=f'Keyword Frequency Comparison ({start_date} to {end_date})',
                          labels={'date': 'Date', 'frequency': 'Total Daily Frequency'})
            # Explicitly set the renderer to 'vscode'.
            # If 'vscode' doesn't work, you can try 'notebook' or other available renderers.
            # You can see available renderers by printing pio.renderers
            fig.show(renderer="vscode")

        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'ngram', and 'total_frequency' columns exist.")
        except Exception as e:
            print(f"Error generating plot: {e}")

    def plot_top_n_grams_trend(self, data: pd.DataFrame, start_date: str, end_date: str, top_n: int = 10) -> None:
        """
        Identifies the top N n-grams within a specified date range based on their total summed frequency,
        and then plots their daily frequency trends over that period.

        Args:
            data (pd.DataFrame): DataFrame with 'date', 'ngram', and 'total_frequency' columns.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            top_n (int): Number of top n-grams to identify and plot. Defaults to 10.
        """
        if top_n <= 0:
            print("Error: top_n must be a positive integer.")
            return
        if top_n > 20: # Limiting to 20 for plot readability, can be adjusted
            print("Warning: Plotting more than 20 n-grams might make the graph cluttered. Consider a smaller top_n.")


        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            # Ensure 'ngram' is string type
            if not pd.api.types.is_string_dtype(data_copy['ngram']):
                data_copy['ngram'] = data_copy['ngram'].astype(str)

            # 1. Filter data by the overall date range
            date_mask_overall = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            period_data = data_copy[date_mask_overall]

            if period_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            # 2. Determine the top N n-grams for the entire period
            top_grams_overall = period_data.groupby('ngram')['total_frequency'].sum().nlargest(top_n).index.tolist()

            if not top_grams_overall:
                print(f"Could not determine top {top_n} n-grams for the period.")
                return

            # 3. For each of these top N n-grams, get their daily frequencies
            all_top_grams_df = pd.DataFrame()

            for ngram_to_plot in top_grams_overall:
                # Filter period_data for the current top n-gram
                ngram_mask = period_data['ngram'] == ngram_to_plot
                daily_data_for_ngram = period_data[ngram_mask]

                if not daily_data_for_ngram.empty:
                    # Group by date and sum total_frequency (should be unique per day for a given ngram already, but sum is safe)
                    daily_freq = daily_data_for_ngram.groupby('date')['total_frequency'].sum().reset_index()
                    daily_freq = daily_freq.rename(columns={'total_frequency': ngram_to_plot})
                    
                    if all_top_grams_df.empty:
                        all_top_grams_df = daily_freq
                    else:
                        all_top_grams_df = pd.merge(all_top_grams_df, daily_freq, on='date', how='outer')
                else:
                    # This case should ideally not happen if ngram_to_plot came from period_data
                    print(f"Note: No daily data found for top n-gram '{ngram_to_plot}' (should not happen).")
            
            if all_top_grams_df.empty:
                print("No daily frequency data found for any of the top n-grams.")
                return
            
            # Fill NaN values that may result from outer merge
            all_top_grams_df = all_top_grams_df.set_index('date').fillna(0).reset_index()
            
            # Melt dataframe for Plotly Express
            plot_df = all_top_grams_df.melt(id_vars=['date'], value_vars=top_grams_overall,
                                           var_name='ngram', value_name='frequency')

            if plot_df.empty:
                print("No frequency data to plot after processing top n-grams.")
                return

            fig = px.line(plot_df, x='date', y='frequency', color='ngram',
                          title=f'Daily Frequency Trend of Top {top_n} n-grams ({start_date} to {end_date})',
                          labels={'date': 'Date', 'frequency': 'Total Daily Frequency', 'ngram': 'N-gram'})
            fig.show(renderer="vscode")

        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'ngram', and 'total_frequency' columns exist.")
        except Exception as e:
            print(f"Error generating top N n-grams trend plot: {e}")

    # --- Hashtag Specific Methods ---

    def top_hashtags_in_date_range(self, data: pd.DataFrame, start_date: str, end_date: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Filter data for top N hashtags within a specified date range,
        based on their summed frequencies. Assumes 'hashtag' and 'total_frequency' columns.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            filtered_data = data_copy.loc[mask]

            if filtered_data.empty:
                return pd.DataFrame(columns=['hashtag', 'total_frequency'])
            
            # Ensure 'hashtag' column exists
            if 'hashtag' not in filtered_data.columns:
                print("Error: 'hashtag' column not found in data.")
                return None
            if 'total_frequency' not in filtered_data.columns:
                print("Error: 'total_frequency' column not found in data.")
                return None

            hashtag_frequencies = filtered_data.groupby('hashtag')['total_frequency'].sum()
            top_hashtags_series = hashtag_frequencies.sort_values(ascending=False).head(top_n)
            top_hashtags_df = top_hashtags_series.reset_index()
            
            return top_hashtags_df
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'hashtag', and 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error filtering and ranking hashtags: {e}")
            return None

    def search_hashtag_in_date_range(self, data: pd.DataFrame, hashtag_to_search: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a hashtag within a specified date range and return its daily occurrences and frequencies.
        Matches hashtag as a whole word, case-insensitively. Assumes 'hashtag' and 'total_frequency' columns.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            
            if 'hashtag' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['hashtag']):
                data_copy['hashtag'] = data_copy['hashtag'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return None


            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            # Using regex for whole word, case-insensitive match on the 'hashtag' column content
            keyword_mask = data_copy['hashtag'].str.contains(f'\\b{hashtag_to_search}\\b', case=False, na=False, regex=True)
            combined_mask = date_mask & keyword_mask
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Hashtag containing '{hashtag_to_search}' not found in the specified date range.")
                return pd.DataFrame(columns=['date', 'hashtag', 'total_frequency'])

            result_df = filtered_data[['date', 'hashtag', 'total_frequency']].sort_values(by='date')
            return result_df
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'hashtag', 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error during hashtag search: {e}")
            return None

    def search_hashtag_total_frequency_in_range(self, data: pd.DataFrame, hashtag_keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a hashtag within a date range and return its total summed frequency.
        Matches hashtag as a whole word, case-insensitively. Assumes 'hashtag' and 'total_frequency' columns.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            if 'hashtag' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['hashtag']):
                data_copy['hashtag'] = data_copy['hashtag'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return None

            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            keyword_mask = data_copy['hashtag'].str.contains(f'\\b{hashtag_keyword}\\b', case=False, na=False, regex=True)
            combined_mask = date_mask & keyword_mask
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Hashtag containing '{hashtag_keyword}' not found in the specified date range.")
                return pd.DataFrame(columns=['hashtag', 'total_frequency_sum'])

            total_freq_sum = filtered_data['total_frequency'].sum()
            results = {'hashtag': hashtag_keyword, 'total_frequency_sum': total_freq_sum}
            return pd.DataFrame([results])
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
            return None
        except Exception as e:
            print(f"Error during hashtag total frequency search: {e}")
            return None

    def plot_hashtag_frequencies_comparison(self, data: pd.DataFrame, hashtags_list: List[str], start_date: str, end_date: str) -> None:
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

        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            
            if 'hashtag' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['hashtag']):
                data_copy['hashtag'] = data_copy['hashtag'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return

            date_mask_overall = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            relevant_data = data_copy[date_mask_overall]

            if relevant_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            all_hashtags_df = pd.DataFrame()

            for ht in hashtags_list:
                keyword_mask = relevant_data['hashtag'].str.contains(f'\\b{ht}\\b', case=False, na=False, regex=True)
                hashtag_data = relevant_data[keyword_mask]

                if not hashtag_data.empty:
                    daily_freq = hashtag_data.groupby('date')['total_frequency'].sum().reset_index()
                    daily_freq = daily_freq.rename(columns={'total_frequency': ht})
                    
                    if all_hashtags_df.empty:
                        all_hashtags_df = daily_freq
                    else:
                        all_hashtags_df = pd.merge(all_hashtags_df, daily_freq, on='date', how='outer')
                else:
                    print(f"Hashtag containing '{ht}' not found in the specified date range.")
            
            if all_hashtags_df.empty:
                print("No data found for any of the specified hashtags in the date range.")
                return

            all_hashtags_df = all_hashtags_df.set_index('date').fillna(0).reset_index()
            plot_df = all_hashtags_df.melt(id_vars=['date'], value_vars=hashtags_list,
                                           var_name='hashtag', value_name='frequency')

            if plot_df.empty:
                print("No frequency data to plot after processing hashtags.")
                return

            fig = px.line(plot_df, x='date', y='frequency', color='hashtag',
                          title=f'Hashtag Frequency Comparison ({start_date} to {end_date})',
                          labels={'date': 'Date', 'frequency': 'Total Daily Frequency', 'hashtag': 'Hashtag'})
            fig.show(renderer="vscode")
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
        except Exception as e:
            print(f"Error generating hashtag comparison plot: {e}")

    def plot_top_hashtags_trend(self, data: pd.DataFrame, start_date: str, end_date: str, top_n: int = 10) -> None:
        """
        Identifies the top N hashtags within a specified date range based on their total summed frequency,
        and then plots their daily frequency trends over that period. Assumes 'hashtag' and 'total_frequency' columns.
        """
        if top_n <= 0:
            print("Error: top_n must be a positive integer.")
            return
        if top_n > 20:
            print("Warning: Plotting more than 20 hashtags might make the graph cluttered.")

        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            if 'hashtag' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['hashtag']):
                data_copy['hashtag'] = data_copy['hashtag'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return

            date_mask_overall = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            period_data = data_copy[date_mask_overall]

            if period_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            top_hashtags_overall = period_data.groupby('hashtag')['total_frequency'].sum().nlargest(top_n).index.tolist()

            if not top_hashtags_overall:
                print(f"Could not determine top {top_n} hashtags for the period.")
                return

            all_top_hashtags_df = pd.DataFrame()

            for ht_to_plot in top_hashtags_overall:
                hashtag_mask = period_data['hashtag'] == ht_to_plot # Exact match for already identified top hashtags
                daily_data_for_hashtag = period_data[hashtag_mask]

                if not daily_data_for_hashtag.empty:
                    daily_freq = daily_data_for_hashtag.groupby('date')['total_frequency'].sum().reset_index()
                    daily_freq = daily_freq.rename(columns={'total_frequency': ht_to_plot})
                    
                    if all_top_hashtags_df.empty:
                        all_top_hashtags_df = daily_freq
                    else:
                        all_top_hashtags_df = pd.merge(all_top_hashtags_df, daily_freq, on='date', how='outer')
            
            if all_top_hashtags_df.empty:
                print("No daily frequency data found for any of the top hashtags.")
                return
            
            all_top_hashtags_df = all_top_hashtags_df.set_index('date').fillna(0).reset_index()
            plot_df = all_top_hashtags_df.melt(id_vars=['date'], value_vars=top_hashtags_overall,
                                               var_name='hashtag', value_name='frequency')

            if plot_df.empty:
                print("No frequency data to plot after processing top hashtags.")
                return

            fig = px.line(plot_df, x='date', y='frequency', color='hashtag',
                          title=f'Daily Frequency Trend of Top {top_n} Hashtags ({start_date} to {end_date})',
                          labels={'date': 'Date', 'frequency': 'Total Daily Frequency', 'hashtag': 'Hashtag'})
            fig.show(renderer="vscode")
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
        except Exception as e:
            print(f"Error generating top hashtags trend plot: {e}")

    # --- Domain Specific Methods ---

    def top_domains_in_date_range(self, data: pd.DataFrame, start_date: str, end_date: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Filter data for top N domains within a specified date range,
        based on their summed frequencies. Assumes 'domain' and 'total_frequency' columns.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            filtered_data = data_copy.loc[mask]

            if filtered_data.empty:
                return pd.DataFrame(columns=['domain', 'total_frequency'])

            if 'domain' not in filtered_data.columns:
                print("Error: 'domain' column not found in data.")
                return None
            if 'total_frequency' not in filtered_data.columns:
                print("Error: 'total_frequency' column not found in data.")
                return None

            domain_frequencies = filtered_data.groupby('domain')['total_frequency'].sum()
            top_domains_series = domain_frequencies.sort_values(ascending=False).head(top_n)
            top_domains_df = top_domains_series.reset_index()
            
            return top_domains_df
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'domain', and 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error filtering and ranking domains: {e}")
            return None

    def search_domain_in_date_range(self, data: pd.DataFrame, domain_to_search: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a domain keyword within a specified date range and return its daily occurrences and frequencies.
        Matches keyword as a whole word, case-insensitively, within the 'domain' column. Assumes 'domain' and 'total_frequency' columns.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            if 'domain' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['domain']):
                data_copy['domain'] = data_copy['domain'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return None

            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            keyword_mask = data_copy['domain'].str.contains(f'\\b{domain_to_search}\\b', case=False, na=False, regex=True)
            combined_mask = date_mask & keyword_mask
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Domain containing '{domain_to_search}' not found in the specified date range.")
                return pd.DataFrame(columns=['date', 'domain', 'total_frequency'])

            result_df = filtered_data[['date', 'domain', 'total_frequency']].sort_values(by='date')
            return result_df
        except KeyError as e:
            print(f"Error processing data: Missing column {e}. Ensure 'date', 'domain', 'total_frequency' columns exist.")
            return None
        except Exception as e:
            print(f"Error during domain search: {e}")
            return None

    def search_domain_total_frequency_in_range(self, data: pd.DataFrame, domain_keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Search for a domain keyword within a date range and return its total summed frequency.
        Matches keyword as a whole word, case-insensitively, within the 'domain' column. Assumes 'domain' and 'total_frequency' columns.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            if 'domain' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['domain']):
                data_copy['domain'] = data_copy['domain'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return None

            date_mask = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            keyword_mask = data_copy['domain'].str.contains(f'\\b{domain_keyword}\\b', case=False, na=False, regex=True)
            combined_mask = date_mask & keyword_mask
            filtered_data = data_copy.loc[combined_mask]

            if filtered_data.empty:
                print(f"Domain containing '{domain_keyword}' not found in the specified date range.")
                return pd.DataFrame(columns=['domain', 'total_frequency_sum'])

            total_freq_sum = filtered_data['total_frequency'].sum()
            results = {'domain': domain_keyword, 'total_frequency_sum': total_freq_sum}
            return pd.DataFrame([results])
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
            return None
        except Exception as e:
            print(f"Error during domain total frequency search: {e}")
            return None

    def plot_domain_frequencies_comparison(self, data: pd.DataFrame, domains_list: List[str], start_date: str, end_date: str) -> None:
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
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            if 'domain' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['domain']):
                data_copy['domain'] = data_copy['domain'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return

            date_mask_overall = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            relevant_data = data_copy[date_mask_overall]

            if relevant_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            all_domains_df = pd.DataFrame()

            for dom in domains_list:
                keyword_mask = relevant_data['domain'].str.contains(f'\\b{dom}\\b', case=False, na=False, regex=True)
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
            fig.show(renderer="vscode")
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
        except Exception as e:
            print(f"Error generating domain comparison plot: {e}")

    def plot_top_domains_trend(self, data: pd.DataFrame, start_date: str, end_date: str, top_n: int = 10) -> None:
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
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            if 'domain' not in data_copy.columns or not pd.api.types.is_string_dtype(data_copy['domain']):
                data_copy['domain'] = data_copy['domain'].astype(str)
            if 'total_frequency' not in data_copy.columns:
                 print("Error: 'total_frequency' column not found.")
                 return

            date_mask_overall = (data_copy['date'] >= start_date) & (data_copy['date'] <= end_date)
            period_data = data_copy[date_mask_overall]

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
            fig.show(renderer="vscode")
        except KeyError as e:
            print(f"Error processing data: Missing column {e}.")
        except Exception as e:
            print(f"Error generating top domains trend plot: {e}")