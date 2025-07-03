import streamlit as st
import requests
import json

st.set_page_config(page_title="Resume Parser", page_icon="📄", layout="wide")

st.title("Resume Parser Tool")
st.markdown("Upload a PDF or DOCX resume to extract key information using AI.")

# File uploader
uploaded_file = st.file_uploader("Choose a resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    with st.spinner("Parsing resume..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://localhost:8000/parse-resume", files=files)
            
            if response.status_code == 200:
                data = response.json()
                
                st.header("Extracted Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Personal Information")
                    st.write(f"**First Name**: {data.get('first_name', 'Not found')}")
                    st.write(f"**Last Name**: {data.get('last_name', 'Not found')}")
                    st.write(f"**Email**: {data.get('email', 'Not found')}")
                    st.write(f"**Phone Number**: {data.get('phone_number', 'Not found')}")
                
                with col2:
                    st.subheader("Professional Details")
                    st.write("**Companies**:")
                    if data.get('companies') and len(data['companies']) > 0:
                        for company in data['companies']:
                            st.write(f"- {company}")
                    else:
                        st.write("Not found")
                    
                    st.write("**Skills**:")
                    if data.get('skills') and len(data['skills']) > 0:
                        for skill in data['skills']:
                            st.write(f"- {skill}")
                    else:
                        st.write("Not found")
                
                st.subheader("Career Goals")
                st.write(data.get('goals', 'Not found'))
                
                st.subheader("Education")
                if data.get('education') and len(data['education']) > 0:
                    for edu in data['education']:
                        st.write(f"- {edu}")
                else:
                    st.write("Not found")
                
                st.subheader("Professional Summary")
                st.write(data.get('summary', 'Not found'))
                
            else:
                st.error(f"Error: {response.json().get('detail', 'Failed to parse resume')}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the backend: {str(e)}. Ensure the backend server is running on http://localhost:8000.")