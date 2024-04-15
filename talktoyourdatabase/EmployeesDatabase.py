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
        and dept_emp on key fields of dept_no and emp_no then count it.
        
        If the input is about salary of employee, then you can look at salaries table on specific date for from_date but if 
        no date was mentioned then you can look at latest year using order by desc. The to_date can be only used for cases like
        identifying when one employee has changes on his salary, for example if input is about: how many times one employee salary
        changes, then you need to query the salaries table using emp_no and check how many rows exist that shows number of salary changes.
    
        If input was about department and salary of department, you need to join three tables of salaries to employees on 
        emp_no field then also join with dept_emp on emp_no to find out which employees work on which department, then you can 
        calculate sum the total salary of each department for all its employees aligning the from_date. To find the department name,
        you can join back with departments on dept_no field.
        
        Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)

mysql_uri = 'mysql+mysqlconnector://root:local@localhost:3306/employees'

engine = create_engine(mysql_uri)
db = SQLDatabase(engine=engine)
print(db.get_usable_table_names())

print("Using local LLM, make sure you have installed Ollama (https://ollama.com/download) and have it running")
llm = Ollama(model="openchat", temperature=0)

agent_executor = create_sql_agent(llm, db=db, verbose=True, max_execution_time=80*60, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,)

with contextlib.suppress(Exception):
    print(agent_executor.invoke(prompt.format(input="how many departments does the company have?"))["output"])
    print(agent_executor.invoke(prompt.format(input="how many employees are there per each department in 1986?"))["output"])
    print(agent_executor.invoke(prompt.format(input="What is the average salary for each department in 1998? "
                                                    "Please show amount and department name."))["output"])
    print(agent_executor.invoke(prompt.format(input="What is my employee turnover rate in 1996?"))["output"])
    print(agent_executor.invoke(prompt.format(input="What is average salary of employees who are older than 25 years?"))["output"])
    print(agent_executor.invoke(prompt.format(input="What is the average salary of employees for Finance department?"))["output"])

