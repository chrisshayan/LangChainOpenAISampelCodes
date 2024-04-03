import pandas as pd

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser


df = pd.read_csv("FinancialStatements20092022.csv")
print(df.shape)
print(df.columns.tolist())


engine = create_engine("sqlite:///FinancialStatements20092022.db")
df.to_sql("FinancialStatements20092022", engine, index=False, if_exists="replace")

db = SQLDatabase(engine=engine)
print(db.dialect)
print(db.get_usable_table_names())

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# agent_executor.invoke({"input": "what's the average revenue of companies in 2022?"})
# Invoking: `sql_db_query_checker` with `{'query': 'SELECT AVG(Revenue) as Average_Revenue FROM FinancialStatements20092022 WHERE Year = 2022'}`
# SELECT AVG(Revenue) as Average_Revenue FROM FinancialStatements20092022 WHERE Year = 2022
# Invoking: `sql_db_query` with `{'query': 'SELECT AVG(Revenue) as Average_Revenue FROM FinancialStatements20092022 WHERE Year = 2022'}`
# [(149006.42545454545,)]The average revenue of companies in 2022 is approximately 149,006.43 million USD.

# agent_executor.invoke({"input": "Can you predict revenue of MSFT in 2023?"})

# agent_executor.invoke({"input": "Which company has a better financial performance from 2020 to 2022?"})
# The table "FinancialStatements20092022" contains financial data for various companies from 2009 to 2022. The relevant columns for determining financial performance could be "Revenue", "Gross Profit", "Net Income", "EBITDA", "Cash Flow from Operating", "ROE", "ROA", "ROI", and "Net Profit Margin".
# However, the question does not specify the criteria for "better financial performance". It could be based on revenue, net income, or any other financial metric.
# For simplicity, let's assume that the company with the highest average net income from 2020 to 2022 has the better financial performance.
# Let's write a SQL query to find the company with the highest average net income from 2020 to 2022.
# SELECT "Company ", AVG("Net Income") as Average_Net_Income FROM FinancialStatements20092022 WHERE Year BETWEEN 2020 AND 2022 GROUP BY "Company " ORDER BY Average_Net_Income DESC LIMIT 1
# Invoking: `sql_db_query` with `{'query': 'SELECT "Company ", AVG("Net Income") as Average_Net_Income FROM FinancialStatements20092022 WHERE Year BETWEEN 2020 AND 2022 GROUP BY "Company " ORDER BY Average_Net_Income DESC LIMIT 1'}`
# [('AAPL', 83964.66666666667)]The company with the best financial performance from 2020 to 2022, based on the highest average net income, is Apple (AAPL) with an average net income of approximately 83,965 million USD.

# agent_executor.invoke({"input": "Which company can afford to take a loan bigger than $1 billion dollar?"})
#The table "FinancialStatements20092022" seems to contain the relevant information. The "Company" column contains the company names, and the "Market Cap(in B USD)" column contains the market capitalization of the companies in billions of dollars.
#A company's ability to afford a loan is typically assessed based on its financial health, which can be indicated by its market capitalization. In this case, we can assume that a company can afford to take a loan bigger than $1 billion if its market capitalization is significantly larger than $1 billion.
#Let's write a query to find companies with a market capitalization larger than $1 billion. We will limit the results to the most recent year available in the data for each company.
#SELECT DISTINCT "Company ", "Market Cap(in B USD)" FROM FinancialStatements20092022 WHERE "Market Cap(in B USD)" > 1 ORDER BY Year DESC LIMIT 10
#Invoking: `sql_db_query` with `{'query': 'SELECT DISTINCT "Company ", "Market Cap(in B USD)" FROM FinancialStatements20092022 WHERE "Market Cap(in B USD)" > 1 ORDER BY Year DESC LIMIT 10'}`
#[('MSFT', 2451.23), ('NVDA', 1000.35), ('AAPL', 2066.94), ('MSFT', 1787.73), ('GOOG', 1144.35), ('PYPL', 81.19), ('AIG', 46.99), ('PCG', 41.28), ('MCD', 186.39), ('BCS', 32.53)]Here are some companies that can afford to take a loan bigger than $1 billion based on their market capitalization:
#1. Microsoft (MSFT) with a market cap of $2451.23 billion
#2. Nvidia (NVDA) with a market cap of $1000.35 billion
#3. Apple (AAPL) with a market cap of $2066.94 billion
#4. Google (GOOG) with a market cap of $1144.35 billion
#5. McDonald's (MCD) with a market cap of $186.39 billion
#Please note that the ability to afford a loan also depends on other factors such as the company's debt levels, cash flow, and profitability, which are not considered in this analysis.

#agent_executor.invoke({"input": "Explain to me what is Current Ratio analysis and why is it important? Also perform it on NVDA and show how you did it."})
#The Current Ratio is a liquidity ratio that measures a company's ability to pay short-term obligations or those due within one year. It compares a firm's current assets to its current liabilities. The formula for calculating the Current Ratio is:
# Current Ratio = Current Assets / Current Liabilities
# A higher Current Ratio indicates the higher capability of a company to pay off its debts. Generally, a Current Ratio of 1.5 to 2 is considered good.
# Now, let's perform the Current Ratio analysis for the company NVDA. Since the database already has a column for Current Ratio, we don't need to calculate it. We just need to query the data for NVDA. Let's do that.
# Error: (sqlite3.OperationalError) no such column: Current_Ratio
# [SQL: SELECT Year, `Company `, Current_Ratio FROM FinancialStatements20092022 WHERE `Company ` = 'NVDA' ORDER BY Year DESC LIMIT 10]
# (Background on this error at: https://sqlalche.me/e/20/e3q8)
# Invoking: `sql_db_query_checker` with `{'query': "SELECT Year, `Company `, `Current Ratio` FROM FinancialStatements20092022 WHERE `Company ` = 'NVDA' ORDER BY Year DESC LIMIT 10"}`
# SELECT Year, `Company `, `Current Ratio` FROM FinancialStatements20092022 WHERE `Company ` = 'NVDA' ORDER BY Year DESC LIMIT 10
# Invoking: `sql_db_query` with `{'query': "SELECT Year, `Company `, `Current Ratio` FROM FinancialStatements20092022 WHERE `Company ` = 'NVDA' ORDER BY Year DESC LIMIT 10"}`
# [(2023, 'NVDA', 3.5156), (2022, 'NVDA', 6.6503), (2021, 'NVDA', 4.0904), (2020, 'NVDA', 7.6738), (2019, 'NVDA', 7.9436), (2018, 'NVDA', 8.0269), (2017, 'NVDA', 4.774), (2016, 'NVDA', 2.5746), (2015, 'NVDA', 6.3761), (2014, 'NVDA', 5.949)]Here are the Current Ratios for NVDA for the last 10 years:

#- 2023: 3.5156
#- 2022: 6.6503
#- 2021: 4.0904
#- 2020: 7.6738
#- 2019: 7.9436
#- 2018: 8.0269
#- 2017: 4.774
#- 2016: 2.5746
#- 2015: 6.3761
#- 2014: 5.949
#These ratios indicate that NVDA has been consistently able to cover its short-term liabilities with its short-term assets, which is a good sign of financial health.

#agent_executor.invoke({"input": "Explain to me what is dupont analysis and why is it important? Also perform it on NVDA and show how you did it."})

#agent_executor.invoke({"input": "With all these financial data available in database, tell me what kind of analysis can you help me with?"})
#1. **Trend Analysis**: Analyze the financial performance of a company over the years. This can include revenue growth, net income growth, changes in market capitalization, etc.
#2. **Ratio Analysis**: Calculate and analyze various financial ratios such as Current Ratio, Debt/Equity Ratio, Return on Equity (ROE), Return on Assets (ROA), Return on Investment (ROI), and Net Profit Margin. These ratios can provide insights into a company's profitability, efficiency, and financial stability.
#3. **Cash Flow Analysis**: Analyze a company's cash flow from operating, investing, and financial activities. This can help understand how a company is generating and using its cash.
#4. **Employee Analysis**: Analyze the number of employees over the years and its relation to the company's financial performance.
#5. **Inflation Impact Analysis**: Analyze the impact of inflation on a company's financial performance.
#6. **Sector Analysis**: Compare the financial performance of companies within the same sector or category.
#7. **Market Capitalization Analysis**: Analyze the changes in a company's market capitalization over the years.

# agent_executor.invoke({"input": "Please do a ratio analysis on NVDA in 2022"})
#The ratio analysis for NVDA in 2022 is as follows:
#- Current Ratio: 6.6503
#- Debt/Equity Ratio: 0.4113
#- Return on Equity (ROE): 36.6451
#- Return on Assets (ROA): 22.0698
#- Return on Investment (ROI): 25.9652
#- Net Profit Margin: 36.2339
#- Free Cash Flow per Share: 1.3378
#- Return on Tangible Equity: 48.946

#agent_executor.invoke({"input": "Please do a sector analysis on NVDA"})
# To perform a sector analysis, we can compare NVDA's financial performance with other companies in the same sector. The relevant columns for this analysis could be "Company", "Category", "Revenue", "Gross Profit", "Net Income", "EBITDA", "ROE", "ROA", "ROI", "Net Profit Margin", and "Number of Employees".
# Comparing these values with the sector averages, we can see that NVDA has lower revenue, gross profit, net income, EBITDA, ROI, and net profit margin than the sector average. However, NVDA's ROE and ROA are comparable to the sector average. Also, NVDA operates with fewer employees than the sector average.

#agent_executor.invoke({"input": "Please do a Market Capitalization Analysis on NVDA"})
#The market capitalization of NVDA over the last 10 years is as follows:
#- 2023: 1000.35 B USD
#- 2022: 359.5 B USD
#- 2021: 735.86 B USD
#- 2020: 323.24 B USD
#- 2019: 144.0 B USD
#- 2018: 81.44 B USD
#- 2017: 117.26 B USD
#- 2016: 57.53 B USD
#- 2015: 17.73 B USD
#- 2014: 10.9 B USD



tool = PythonAstREPLTool(locals={"df": df})
llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
llm_with_tools.invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'EBITDA' and 'Earning Per Share'"
)

parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)
(llm_with_tools | parser).invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'EBITDA' and 'Earning Per Share'"
)

print(parser)

system = f"""You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:

```
{df.head().to_markdown()}
```

Given a user question, write the Python code to answer it. \
Return ONLY the valid Python code and nothing else. \
Don't assume you have access to any libraries other than built-in Python ones and pandas."""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
code_chain = prompt | llm_with_tools | parser

# code_chain.invoke({"question": "What's the correlation between the 'EBITDA' and 'Earning Per Share'"})
# 0.1789879025071459

#chain = prompt | llm_with_tools | parser | tool  # noqa
# print(chain.invoke({"question": "What's the correlation between the 'Revenue' and 'Gross Profit'"}))
# 0.9471892084246643

# print(code_chain.invoke({"question": "Can you summarize the performance of AMZN and compare it to LOGI from 2010 to 2020?"}))
