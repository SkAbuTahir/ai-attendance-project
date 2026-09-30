import streamlit as st
from src.database.db import enroll_student_to_subject, get_all_subjects
from src.database.config import supabase
import time


@st.dialog("Enroll in Subject")
def enroll_dialog():
    student_id = st.session_state.student_data['student_id']

    # fetch already enrolled subject ids to exclude them
    enrolled_res = supabase.table('subject_students').select('subject_id').eq('student_id', student_id).execute()
    enrolled_ids = {row['subject_id'] for row in enrolled_res.data}

    all_subjects = get_all_subjects()
    available = [s for s in all_subjects if s['subject_id'] not in enrolled_ids]

    if not available:
        st.info('You are already enrolled in all available subjects.')
        return

    # build display labels
    options = {
        f"{s['name']} — {s['subject_code']} | Section {s['section']} | 👨‍🏫 {s.get('teachers', {}).get('name', 'N/A')}": s
        for s in available
    }

    st.markdown('#### Select a subject to enroll in')
    selected_label = st.selectbox(
        'Available Subjects',
        options=list(options.keys()),
        index=0,
        label_visibility='collapsed'
    )

    selected = options[selected_label]

    with st.container(border=True):
        st.markdown(f"**{selected['name']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric('Code', selected['subject_code'])
        c2.metric('Section', selected['section'])
        c3.metric('Teacher', selected.get('teachers', {}).get('name', 'N/A'))

    st.divider()

    # also allow manual code entry as fallback
    with st.expander('Have a join code instead?'):
        join_code = st.text_input('Subject Code', placeholder='Eg. CS101')
        if st.button('Join by code', type='secondary', width='stretch'):
            if join_code:
                res = supabase.table('subjects').select('subject_id, name').eq('subject_code', join_code).execute()
                if res.data:
                    subject = res.data[0]
                    if subject['subject_id'] in enrolled_ids:
                        st.warning('You are already enrolled in this subject.')
                    else:
                        enroll_student_to_subject(student_id, subject['subject_id'])
                        st.success(f"Enrolled in {subject['name']}!")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error('Subject code not found.')
            else:
                st.warning('Please enter a subject code.')

    if st.button('Enroll Now', type='primary', width='stretch'):
        enroll_student_to_subject(student_id, selected['subject_id'])
        st.success(f"Successfully enrolled in **{selected['name']}**!")
        time.sleep(1)
        st.rerun()
