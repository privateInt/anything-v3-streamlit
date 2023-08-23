from utils import load_model, inference, txt_to_img, img_to_img, gen_models, gen_scheduler
import streamlit as st

def main():
    device = 'cuda:1'
    
    models = gen_models()
    scheduler = gen_scheduler()
    
    with st.form('anything'):
        st.title("anything-v3.0 test web page")
        
        choose_model = st.selectbox('if you want to use other model, choose other model', [m.name for m in models])
        for model in models:
            if choose_model == model.name:
                current_model = model
        last_mode = "txt2img"
        model = load_model(current_model, scheduler)
        
        prompt = st.text_input(label="Prompt", placeholder="Enter prompt. Style applied automatically")
        neg_prompt = st.text_input(label="Negative prompt", placeholder="What to exclude from the image")
        guidance = st.slider("Guidance scale", min_value = 0.0, max_value = 15.0, value = 7.5)
        steps = st.slider("Steps", 2, 75, 25, 1)
        width = st.slider("Width", 64, 1024, 512, 8)
        height = st.slider("Height", 64, 1024, 512, 8)
        seed = st.slider("Seed (0 = random)", 0, 214748, 0, 1) # max_value=2147483647
        add_selectbox = st.sidebar.selectbox("if you want to use image to image, choose Image to Image", ("Text to Image", "Image to Image"))
        if add_selectbox == "Image to Image":
            last_mode = "img2img"
            img = st.file_uploader("drop image here, if you want to use image to image")
            strength = st.slider("select Transformation strength,if you want to use image to image", 0.0, 1.0, 0.5, 0.01)
        elif add_selectbox == "Text to Image":
            last_mode = "txt2img"
            
        if st.form_submit_button(label='Submit'):
            if last_mode == "txt2img":
                rslt = inference(model, last_mode, prompt, guidance, steps, device, width, height, seed, neg_prompt=neg_prompt)
                st.image(rslt.images[0])
            else:
                rslt = inference(model, last_mode, prompt, guidance, steps, device, width, height, seed, img, strength, neg_prompt=neg_prompt)
                st.image(rslt.images[0])

if __name__ == '__main__':
    main()
