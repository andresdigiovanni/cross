import os


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    server_app_path = os.path.join(project_root, "..", "applications", "server.py")

    os.system("streamlit run {}".format(server_app_path))


if __name__ == "__main__":
    main()
