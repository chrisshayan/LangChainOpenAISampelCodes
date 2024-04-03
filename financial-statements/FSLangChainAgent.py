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
print(db.run("SELECT * FROM FinancialStatements20092022 where Revenue > 394328;"))

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

agent_executor.invoke({"input": "Which company can afford to take a loan bigger than $1 billion dollar?"})
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
