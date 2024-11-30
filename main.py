import streamlit as st
import pickle
import spam_email

model = pickle.load(open('spam123.pkl','rb'))
cv = pickle.load(open('vec123.pkl','rb'))

def main():
    st.title("EMAIL SPAM CLASSIFICATION APPLICATION.")
    st.write("This is a Machine Learning Application created by PRADEEP SINGH to classify emails as spam or ham.")
    st.subheader("Classification!")
    user_input = st.text_area("Enter an email to classify", height=100)
    if st.button("Classify"):
        if user_input:
            data = [user_input]
            print(data)
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == 0:
                st.success("This is  Not A SPAM EMAIL.")
            else:
                st.error("This is A SPAM EMAIL.")
        else:
            st.write("Please enter an Email to Classify.")
main()