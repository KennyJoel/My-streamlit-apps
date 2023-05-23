import streamlit as st

def main():
    st.set_page_config(page_title="Kenny Joel's Webpage", page_icon=":smiley:")

    # Add CSS styles
    st.markdown(
        """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 2rem;
        }
        h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }
        h2 {
            color: #555;
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        p {
            color: #777;
            font-size: 1.2rem;
        }
        .profile-picture {
            max-width: 200px;
            border-radius: 50%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add a title and description
    st.title("Kenny Joel's Webpage")
    st.markdown("Welcome to my personal webpage!")

    # Add some content
    st.header("About Me")
    st.write("I am a passionate developer with expertise in web development and data analysis.")

    st.header("Skills")
    st.write("Here are some of my key skills:")
    st.write("- Web development (HTML, CSS, JavaScript)")
    st.write("- Python programming")
    st.write("- Data analysis and visualization")

    st.header("Projects")
    st.write("I have worked on several interesting projects, including:")
    st.write("- Building a web-based expense tracker")
    st.write("- Analyzing customer behavior data")

    st.header("Contact")
    st.write("Feel free to reach out to me via email: kennyjoel@example.com")

    # Add some visual elements
    st.header("Profile Picture")
    st.image("profile_picture.jpg", caption="Kenny Joel", className="profile-picture")

    st.header("Social Media")
    st.markdown(
        """
        You can find me on:
        - [LinkedIn](https://www.linkedin.com/in/kuragayala-kenny-joel/)
        """
    )

if __name__ == "__main__":
    main()
