import os
import re
import torch
import shutil
import streamlit as st

from hloc.utils import viz_3d
import firebase_admin
from firebase_admin import credentials, auth
from utils.sfm_pipeline import PipelineManager

feat_map = {'LightGlue+Aliked':'aliked',
            'LightGlue+SuperPoint':'superpoint',
            'LightGlue+DoGHardNet':'doghardnet',
            'LightGlue+DISK':'disk', 
            'LightGlue+SIFT':'sift'}

# @st.cache_resource
# def initialize_firebase(creds_path, options):
#     creds = credentials.Certificate(creds_path)
#     firebase_admin.initialize_app(creds, options)
#     print('Firebase application initialized successfully!')

class ProjectUtilities:
    def __init__(self, folder_name='recon', parent_dir='outputs'):
        self.project_path = f'{parent_dir}/{folder_name}_project'
        self.images_path = f'{self.project_path}/images'
        
        # Create project directory
        try:
            os.makedirs(self.project_path, exist_ok=True)
            os.makedirs(self.images_path, exist_ok=True)
            if 'project_path' not in st.session_state:
                st.session_state['project_path'] = self.project_path
            if 'images_path' not in st.session_state:
                st.session_state['images_path'] = self.images_path

        except PermissionError as e:
            st.error(f'Permission denied: {e}')
        except Exception as e:
            st.error(f'Error creating directory: {e}')
    
    def save_image(self, image):
        try:
            with open(os.path.join(self.images_path, image.name), 'wb') as f:
                f.write(image.getbuffer())
            return self.images_path
        except Exception as e:
            st.error(f'Error saving file: {e}')
            return None
    
    def delete_image_folder(self):
        try:
            if os.path.exists(self.images_path):
                shutil.rmtree(self.images_path)
            else:
                st.error(f'{self.images_path} does not exist.')
        except Exception as e:
            st.error(f'Error deleting folder: {e}')

class Page:
    def __init__(self):
        self.components = {}

    def add_component(self, id, component):
        self.components[id] = component
    
    def render(self):
        for id, component in self.components.items():
            component.render()

# class UserAuthenticationComponent:
#     def __init__(self):
#         self.email = ''
#         self.user = ''
#         self.password = ''

#     def validate_email(self, email):
#         regex = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
#         return re.match(regex, email)

#     def check_fields(self, email, password):
#         if email == '' or password == '':
#             st.error('Email or password is missing.')
#             return False
        
#         if not self.validate_email(email):
#             st.error('Invalid Email Address.')
#             return False
        
#         return True

#     def login(self):
#         try:
#             self.user = auth.get_user_by_email(self.email) 
#             st.session_state['user'] = self.user.email
#             st.session_state['user_id'] = self.user.uid

#             st.success(f'Logged in successfully.')
#         except auth.UserNotFoundError:
#             st.error('User not found. Select "Register" option from the dropdown to create a new account.')
#         except Exception as e:
#             st.error(f'Error: {e}')
    
#     def register(self):
#         try:
#             auth.create_user(email = self.email, password = self.password)
#             st.success('Registered successfully! Please log in.')
#         except Exception as e:
#             st.error(f'Error: {e}')
    
#     def render(self):
#         st.divider()
#         option = st.selectbox('Login or Register', ['Login', 'Register'])
#         if option == 'Login':
#             with st.form(key = option):
#                 self.email = st.text_input('Email')
#                 self.password = st.text_input('Password', type = 'password')
#                 submit = st.form_submit_button(label = 'Log In')
        
#                 if submit:
#                     if self.check_fields(self.email, self.password):
#                         self.login()

#         if option == 'Register':
#             with st.form(key = option):
#                 self.email = st.text_input('Email')
#                 self.password = st.text_input('Password', type = 'password')
#                 submit = st.form_submit_button(label = 'Create account')

#                 if submit:
#                     if self.check_fields(self.email, self.password):
#                         self.register()    

# class AccountInfoComponent:
#     def render(self):
#         st.divider()
#         st.subheader('Account Information')
#         with st.container(border = True):
#             first_row, second_row = st.columns(2), st.columns(2)
#             first_row[0].container().write(f"Logged in as \n\n > 🟢 **{st.session_state['user']}**")
#             first_row[1].container().write(f"User ID\n\n > {st.session_state['user_id']}")

#             with second_row[0].container():
#                 if st.button('Logout', use_container_width = True):
#                     st.session_state['user'] = None
#                     st.success('Logged out successfully')
            
#             with second_row[1].container():
#                 st.button('Upgrade', use_container_width = True, disabled = True)

class ImagePreviewComponent:
    def __init__(self, project:ProjectUtilities, max_slots = 5, file_formats = ['jpg', 'png', 'jpeg']):
        self.project = project
        self.max_slots = max_slots
        self.file_formats = file_formats

        if 'images' not in st.session_state:
            st.session_state['images'] = []

    def render(self):
        st.divider()      
        st.subheader('Upload Images')
        uploaded_images = ''
        with st.container(border = True):
            with st.form('uploader', clear_on_submit = True):
                uploaded_images = st.file_uploader(label = '---', 
                                                accept_multiple_files = True,
                                                type = self.file_formats)
                save = st.form_submit_button('Upload Images', use_container_width = True)
                if save:
                    for img in uploaded_images:
                        self.project.save_image(img)

            rows = st.columns(2)
            with rows[0]:
                if st.button('Reset', use_container_width = True):
                    self.project.delete_image_folder()
            
            with rows[1]:
                preview_btn = st.button(f'Preview Images ({self.max_slots})', use_container_width = True)
            if preview_btn:
                n_slots = len(uploaded_images) if len(uploaded_images) < self.max_slots else self.max_slots
                img_slots = st.columns(n_slots)
                for idx, image in enumerate(uploaded_images[:n_slots]):
                    img_slots[idx].image(image)

class ReconstructionComponent:
    def __init__(self):
        self.rec = ''
        self.rec_fig = ''
    def render(self):
        st.divider()
        st.subheader('Generate Reconstruction')
        with st.container(border = True):
            rows = st.columns(2)
            feature_matchers = rows[0].multiselect('---',
                                                    list(feat_map.keys()),
                                                    ['LightGlue+Aliked'])
            rows[1].write('')
            rows[1].write('')
            gpu = rows[1].toggle('Enable GPU')
            start_rec = st.button('Start Reconstruction', use_container_width = True)
        
        if start_rec:
            if not feature_matchers:
                st.error('Please Select a Feature Matcher.')
            else:
                if os.path.exists(st.session_state['project_path'] + '\images_rec'):
                    shutil.rmtree(st.session_state['project_path'] + '\images_rec')
                
                if os.path.exists('colmap_rec.zip'):
                    os.remove('colmap_rec.zip')

                pipe_manager = PipelineManager(img_dir = st.session_state['images_path'],
                                                device = torch.device('cuda') if gpu else torch.device('cpu'))

                for idx, fm in enumerate(feature_matchers):
                    pipe_manager.create_instance(instance_id = idx, 
                                                    model_name = feat_map[fm])
                
                with st.status('Running Reconstruction'):
                    self.rec, output_path = pipe_manager.run_reconstruction_for_all()
                    self.rec_fig = viz_3d.init_figure(height = 500)
                    viz_3d.plot_reconstruction(self.rec_fig, self.rec, cameras = False, 
                                        color = 'rgba(227,168,30,0.5)', cs = 5)
                st.plotly_chart(self.rec_fig)
            
                shutil.make_archive('colmap_rec', 'zip', output_path)
                download_btn = st.download_button(label = 'Download Reconstruction Zip File',
                                        data = open('colmap_rec.zip', 'rb'), 
                                        file_name = 'colmap_rec.zip', 
                                        mime = 'application/zip', 
                                        use_container_width = True)
def initialize_app():
    initialize_firebase(**config)

    if 'user' not in st.session_state and \
        'user_id' not in st.session_state:

        st.session_state['user'] = None
        st.session_state['user_id'] = None

def main():
    st.title('Structure-from-Motion with LightGlue')
    st.markdown("""
    This application allows users to upload their own Structure-from-Motion (SfM) datasets to create point-cloud reconstructions using **LightGlue**.

    **LightGlue**: A lightweight feature matcher known for its high accuracy and rapid inference. 
        LightGlue uses adaptive pruning techniques for both network width and depth, taking a set of keypoints and descriptors from each image and 
        returning the indices of corresponding points. [GitHub](https://github.com/cvg/LightGlue?tab=readme-ov-file)""")

    # initialize_app()
    # AuthPage = Page()
    SfMPage = Page()

    # userauth = UserAuthenticationComponent()
    # AuthPage.add_component('Authentication', userauth)

    # if st.session_state['user']:
    project = ProjectUtilities()
    # SfMPage.add_component('SfM', AccountInfoComponent())
    SfMPage.add_component('ImagePreview', ImagePreviewComponent(project = project))
    SfMPage.add_component('Reconstruction', ReconstructionComponent())

    # if st.session_state['user']:
    #     SfMPage.render()
    # else:
    #     AuthPage.render()

    SfMPage.render()

if __name__ == '__main__':
    main()
