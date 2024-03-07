import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import os

st.set_page_config(page_title="LLM for Testing - CrewAI - Backend Test Case Writing", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>            
.stButton > button {
    width: 100%;
    height: 50px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Write Test Cases for Backend</h1>", unsafe_allow_html=True)


if st.button('Back To Backend Testing', key='back_to_backend', type="secondary", help="Click for Backend Testing"):
    st.switch_page("pages/2_Backend_Testing.py")


st.write("")
st.write("")
st.write("")

api_name = st.text_input(type="default", label="API Name", help="Please Enter the Name of the API Endpoint here.")
http_method = st.text_input(type="default", label="HTTP Method of API", help="Please Enter the HTTP method of the API Ex: POST, GET, DELETE, PUT etc.")
type_of_end_users = st.text_input(type="default", label="Type of End Users", help="Enter the type of the End Users as per their roles." )
main_objective = st.text_area(label="Main Business  Objective of API", help="Enter the Primary Business Objective to be tested.")
sub_objective = st.text_area(label="Sub Business Objectives of API", help="Enter Sub Business Objectiives to be tested. These objectives should be secondary objectives than the Primary Objective.")
mandatory_header_parameters = st.text_input(type="default", label="Mandatory Header Parameters")
non_mandatory_header_parameters = st.text_input(type="default", label="Non-Mandatory Header Parameters")
mandatory_request_parameters = st.text_input(type="default", label="Mandatory Request Parameters")
non_mandatory_request_parameters = st.text_input(type="default", label="Non-Mandatory Request Parameters")
data_transfer_method = st.selectbox('Data Transfer Method', ('Request Body', 'Path Variable', 'Query Parameter'))

st.write("")
st.write("")
st.write("")

generate_test_cases_btn = st.button('Generate Test Cases', key='generate_test_cases_btn', type="primary", use_container_width=True, help="Click for Write Test Cases")

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LLM for Testing Application Crew AI - Write Backend Test Cases"

def create_test_cases(test_scenario):
    quality_assuarance_engineer = Agent(
        role='Expert Senior Quality Assuarance Engineer',
        goal='Write the all possible positive and negative test cases',
        verbose=True,
        backstory="""
        You are Expert Quality Assuarance Engineer, and specilist for write backend test cases
        Your responsibility is write the possible poitive and negative test cases and expected results with covering all points for given test scenario.
        When writing the test cases, follow  QA standards, and keywords.
        Write the test cases according to test cases document written style.
        """,
        memory=True
    )

    write_positive_test_cases = Task(
        description=f"""
            This is the Scenario. I want to write the all positive test cases for this scenario. 

            {test_scenario}
        
            Think this above scenario step by step, and consider each and every point.
            write the all possible positive test cases for this scenario.
            When you write the test cases, cover all possible positive test cases.
            Don't miss any of the positive test case, try to cover every positive test cases for this scenario.
            When writing the test cases, follow  QA standards, and keywords (use 'verify' keyword if want).
        """,
        expected_output="""
            All possible positive test cases list according to below template.

            Positive test cases :
            1. Test Case : << When QA engineer write the test cases, they follow the commen pattern for write single test case. You also need follow that pattern for write the test case >>
                Test Data : << You need to write test data according to test case document >>
                Expected Output : << You need to write expected output according to test case document >>
            2. Test Case : << When QA engineer write the test cases, they follow the commen pattern for write single test case. You also need follow that pattern for write the test case >>
                Test Data : << You need to write test data according to test case document >>
                Expected Output : << You need to write expected output according to test case document >>
        """,
        agent=quality_assuarance_engineer
    )

    write_negative_test_cases = Task(
        description=f"""
            This is the Scenario. I want to write the all negative test cases for this scenario. 

            {test_scenario}
        
            Think this above scenario step by step, and consider each and every point.
            Write the all possible negative test cases for this scenario.
            When you write the test cases, cover all possible negative test cases.
            Don't miss any of the negative test case, try to cover every negative test cases for this scenario.
            When writing the test cases, follow  QA standards, and keywords (use 'verify' keyword if want). 
        """,
        expected_output="""
            All posible negative test cases list according to below template.

            Negative test cases :
            1. Test Case : << When QA engineer write the test cases, they follow the commen pattern for write single test case. You also need follow that pattern for write the test case >>
                Test Data : << You need to write test data according to test case document >>
                Expected Output : << You need to write expected output according to test case document >>
            2. Test Case : << When QA engineer write the test cases, they follow the commen pattern for write single test case. You also need follow that pattern for write the test case >>
                Test Data : << You need to write test data according to test case document >>
                Expected Output : << You need to write expected output according to test case document >>
        """,
        agent=quality_assuarance_engineer
    )

    combine_all_test_cases = Task(
        description=f"""
        After write the all possible positive and negative test cases, combine all the positive and negative test cases together.
        Don't miss any of the test cases. give all genarated positive and negative test cases.      
        """,
        expected_output="""
            All possible positive and negative test cases list according to below template.

            Positive Test Cases :
                1. 
                    Test Case -
                    Test Data -
                    Expected Result -
                2. 
                    Test Case -
                    Test Data -
                    Expected Result -

            Negative Test Cases :
                1. 
                    Test Case -
                    Test Data -
                    Expected Result -
                2. 
                    Test Case -
                    Test Data -
                    Expected Result -
        """,
        context=[write_positive_test_cases, write_negative_test_cases],
        agent=quality_assuarance_engineer
    )


    crew = Crew(
        agents=[quality_assuarance_engineer],
        tasks=[write_positive_test_cases, write_negative_test_cases, combine_all_test_cases],
        verbose=True,
        manager_llm=ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.0),
        process=Process.sequential
    )

    results = crew.kickoff()
    return results


st.write("")
st.write("")

if generate_test_cases_btn:
    with st.spinner('Please wait until the test cases are generated'):
        if not api_name or not http_method or not type_of_end_users or not main_objective or not sub_objective or not mandatory_header_parameters or not non_mandatory_header_parameters or not mandatory_request_parameters or not non_mandatory_request_parameters or not data_transfer_method:
            error_msg = """
            Please Fill the All input fields. Fields cannot be empty. If you have not value for particular field, Then put 'N/A' in that field.
            """
            st.error(error_msg)
        else:
            test_scenario = f"""
                Scenario : \n
                    API Name - {api_name} \n
                    HTTP Method of API - {http_method} \n
                    Type of End Users - {type_of_end_users} \n
                    Main Business  Objective of API - {main_objective} \n
                    Sub Business Objectives of API - {sub_objective} \n
                    Mandatory Header Parameters - {mandatory_header_parameters} \n
                    Non-Mandatory Header Parameters - {non_mandatory_header_parameters} \n
                    Mandatory Request Payload Parameters - {mandatory_request_parameters} \n
                    Non-Mandatory Request Payload Parameters - {non_mandatory_request_parameters} \n
                    Data Transfer Method - {data_transfer_method} \n
            """
            # result = create_test_cases(test_scenario)
            st.text_area("Generated Test Cases", value=test_scenario, height=1000, key='test_cases_list')


