import pandas as pd

def load_login_data(logindata_filepath):
    #reading in the login data
    #passing a list of datatypes for the columns
    login_dtypes = {"ID": int, "RES_ID": int, "DATE_SAVED": str, "DATE_LAST_MODIFIED": str, "CUST_CODE": str, 
                "IS_ARRIVED": str, "LOCATION": int, "COMPUTED_RSV_ID": str}

    #columns that are given as dates can be parsed while reading 
    login_date_columns = ['DATE_SAVED', 'DATE_LAST_MODIFIED']

    login_dataframe = pd.read_csv(logindata_filepath, dtype=login_dtypes, parse_dates=login_date_columns)

    return login_dataframe

def load_customer_data(userdata_filepath):
    #reading in the customer data
    #passing a list of datatypes for the columns of customer base
    customer_dtypes = {'ISACTIVE': bool, "CUST_CODE": str, 'CLASS_CODE': str, 'BUSINESS_CODE': str, 'SEX': str, 'NO_MAIL': str, 
                    'NO_SMS': str, 'COMPANY_CUST_CODE': str, 'IS_DROP_IN_CUSTOMER': str}

    #list of columns that will be parsed as dates
    customer_date_cols = ['EXPIRE_DATE', 'LAST_VISIT', 'DATE_SAVED', 'DATE_LAST_MODIFIED', 'RENEW_DATE', 'PAUSE_START', 'PAUSE_END', 'MEMBER_SINCE']

    customer_base = pd.read_csv(userdata_filepath, dtype=customer_dtypes, parse_dates=customer_date_cols, index_col=0)

    return customer_base

def load_result_data(result_filepath):
    """
    Helper routine for reading in result files
    """

    result_data = pd.read_csv(result_filepath, compression="gzip", index_col=0, dtype={"CUST_CODE": str})

    return result_data