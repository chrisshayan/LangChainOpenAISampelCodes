from langchain.agents import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import contextlib


template = """
        Role: AI Assistant Expert 
        Department: HR, CEO Office 
        
        Primary Responsibility: You are a helpful AI assistant expert in identifying the relevant topic from user's question 
         about Departments, Employees, Employee's salary, department's manager, job titles and employee's titles and 
         then querying SQL Database to find answer.
        
        Use following context to create the SQL query. 
            Context: 
                departments table includes information about company's departments such as a unique identifier 
                as dept_no and department name dept_name. If the input is about number of departments then count at departments table

                employees table includes information about company's employees such as  emp_no as a unique identifier,
                birth_date birthday of employee, first_name & last_name employee's name, gender and hire_date employee's
                recruitment date.
                
                dept_emp table includes information on which employee (emp_no) works for which department (dept_no) 
                on specific time period (from_date to to_date). 
                
                salaries table Tracks employee salaries (emp_no, from_date, salary). 
                For example it says which employee emp_no from when from_date, how much salary he is earning based on salary field. 
                If to_date has value it could mean user either has another record to show a new salary which can be meaning
                increase or decrease in salary. You can assume if there is any record for an employee, that employee 
                has got paid if  date is valid.
                
                You have access to all MySQL commands such as LAG, DESC.
        
        If the input is about number of employees of each department, then join three tables of departments, employees 
        and dept_emp on key fields of dept_no and emp_no then count it
        
        Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)

mysql_uri = 'mysql+mysqlconnector://root:local@localhost:3306/employees'

engine = create_engine(mysql_uri)
db = SQLDatabase(engine=engine)
print(db.get_usable_table_names())

print("Using local LLM, make sure you have installed Ollama (https://ollama.com/download) and have it running")
llm = Ollama(model="openchat", temperature=0)

agent_executor = create_sql_agent(llm, db=db, verbose=False, max_execution_time=10*60, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,)

with contextlib.suppress(Exception):
    #print(agent_executor.invoke(prompt.format(input="Please summarize what you see interesting in max 300 words."))["output"])
    # print(agent_executor.invoke(prompt.format(input="how many employees are there per each department in 2006?"))["output"])

    print(agent_executor.invoke(prompt.format(input="how many departments does the company have?"))["output"])
    print(agent_executor.invoke(prompt.format(input="What is the average salary of employees for each department in 2002? "
                                                    "Please show amount and department name."))["output"])
    print(agent_executor.invoke(prompt.format(input="What is average salary of employees who are older than 25 years?"))["output"])
    print(agent_executor.invoke(prompt.format(input="How much total salary increased comparing 1995 to 2002?"))["output"])
    print(agent_executor.invoke(prompt.format(input="which department is the highest salary?"))["output"])
    #print(agent_executor.invoke(prompt.format(input="Which employee (employee id and full name) has highest salary in 2002?"))["output"])
    #print(agent_executor.invoke(prompt.format(input="How many times employee number 10001 got salary increase?"))["output"])
    #print(agent_executor.invoke(prompt.format(input="What is the average salary of employees for Finance department?"))["output"])

