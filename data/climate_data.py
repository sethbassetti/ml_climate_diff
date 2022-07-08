from torch.utils.data import Dataset, DataLoader
import torch
import random
import tqdm

PR_CONSTANT = 86400
DATA_START = 1850
DATA_END = 2100

class ClimateDataset(Dataset):
    def __init__(self, dataset_path=None, variable=None, seq_len=32, year_bounds=None):
        """Initializes class variables and loads the data into memory"""

        self.seq_len = seq_len
        self.variable = variable

        # End year is not inclusive
        self.start_year = year_bounds[0]
        self.end_year = year_bounds[1]

        # Dataset will be a tensor in the shape of #num_realizations x num_days x height x width
        self.dataset = torch.load(dataset_path)

        # If year bounds are given, then only use the data within those year bounds
        if year_bounds:
            self.dataset = self.extract_by_years(self.start_year, self.end_year)

        self.num_realizations = self.dataset.shape[0]

        # This is so that, for example, to get a chunk of 32 days, it doesn't attempt to start on the last day or days
        self.days_per_realization = self.dataset.shape[1] - (self.seq_len-1)


    def extract_by_years(self, start_year, end_year):
        """Slices a portion of the dataset based on a start year and end year"""

        # Make sure the start year and end year are within bounds
        assert start_year >= DATA_START, "Starting year is before dataset begins!!"
        assert end_year <= DATA_END, "End year is after dataset ends!"

        # Calculate the total years and days of this interval
        total_years = (end_year + 1) - start_year
        total_days = total_years * 365

        # Calculate the num of years and number of days to start the interval past the start of the dataset
        year_offset = start_year - DATA_START
        day_offset = year_offset * 365

        return self.dataset[:, day_offset:day_offset+total_days]



    def preprocess(self, month, cond_month):

        # If this is precipitation, convert units to mm/day using the PR_CONSTANT
        if self.variable == 'pr':
            processed_month = month * PR_CONSTANT
            processed_cond_month = cond_month * PR_CONSTANT
        else:
            processed_month = month
            processed_cond_month = cond_month

        processed_month = processed_month.unsqueeze(0)
        processed_cond_month = processed_cond_month.unsqueeze(0)

        return processed_month, processed_cond_month

    def get_resolution(self):
        """Returns the shape of the dataset"""
        return tuple(self.dataset.shape[-2:])

    def create_var_map(self):
        return {self.variable:0}

    def estimate_num_batches(self, batch_size, drop_last):
        return len(self) // batch_size - int(drop_last)

    def calc_year_day(self, idx):
        """Given an index, calculates what year and what day this sample starts from"""
        
        # Since our idx's measure cannot start in the last self.seq_len-1 days
        elapsed_days = idx % self.days_per_realization
        elapsed_years = elapsed_days // 365

        current_day = elapsed_days % 365

        current_year = self.start_year + elapsed_years
        return current_year, current_day


    def __getitem__(self, idx):
        """Given an index, constructs an appropriate month-length sequence from the dataset. Additionally, selects
        a random sequence from the data to be used for conditioning."""

        # This gets the day to start our sequence at
        
        realization_idx = idx // self.days_per_realization
        day_idx = idx - realization_idx*self.days_per_realization

        # Selects an appropriately sized month-chunk from the realization
        realization = self.dataset[realization_idx]
        month = realization[day_idx:day_idx + self.seq_len]

        # Conditioning selects a random realization and a random day to start the month sequence within realization
        rand_idx = random.randint(0, self.num_realizations-1)
        cond_realization = self.dataset[rand_idx]

        # Make sure to only select up to the last seq_len days so that you do not go out of index on the dataset
        cond_day_start = random.randint(0, self.days_per_realization-1)
        cond_month = cond_realization[cond_day_start:cond_day_start+self.seq_len]

        # Performs preprocessing on both tensors before returning them to user
        processed_month, processed_cond = self.preprocess(month, cond_month)

        sample_year, sample_day = self.calc_year_day(idx)
        cond_year, cond_day = self.calc_year_day(rand_idx * self.days_per_realization + cond_day_start)
        
        return processed_month, sample_year, sample_day, processed_cond, cond_year, cond_day

    def __len__(self):
        """Returns the length of the dataset, which will be the number of month-chunks among every realization"""


        # This is month-chunks per realization multiplied by number of realizations
        num_months = self.days_per_realization * self.num_realizations
        return num_months
        
        


def new_climate_dataloader(dataset, desc='', is_last=False, disable_tqdm=False, drop_last=True, batch_size=16, **dataloader_kwargs):
    """Creates a generator that will yield datapoints from ClimateData object

    Arguments:
        - dataset (ClimateData)    -- The dataset class
        - desc (str)               -- Prefix for tqdm iterator
        - is_last (bool)           -- If this is the last epoch to be yielded
        - disable_tqdm (bool)      -- Disables the progress bar
        - drop_last (bool)         -- If the last batch should be dropped if its not equal to batch_size
        - batch_size (int)         -- Number of samples per batch
        - dataloader_kwargs (dict) -- All additional kwargs for the pytorch dataloader
    """
    # Estimate the number of batches for the progress bar
    num_batches = dataset.estimate_num_batches(batch_size, drop_last)
    t = tqdm.tqdm(total=num_batches, desc=desc, disable=disable_tqdm)
    batch_index = 0


    # Create a new pytorch dataloader to handle this chunk
    dl = DataLoader(dataset, drop_last=drop_last, batch_size=batch_size, **dataloader_kwargs)

    # Yield each sample and update our progress bar
    for sample in dl:
        yield sample
        batch_index += 1
        t.update()
