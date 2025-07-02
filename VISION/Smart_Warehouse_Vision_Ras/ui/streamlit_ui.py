# ui/streamlit_ui.py
"""
Streamlit User Interface Module - Simplified Demo Version
Contains the StreamlitUI class responsible for all user interface components and interactions.
"""

import json
import os
import tempfile
import time
from datetime import datetime
from typing import Dict

import cv2
import pandas as pd
import streamlit as st

from config.warehouse_config import WarehouseConfig
from core.video_processor import VideoProcessor


class StreamlitUI:
    """
    Handles all Streamlit user interface components including configuration, 
    session management, live display, and confirmation workflows.
    """
    
    def __init__(self):
        """Initialize UI with session state management."""
        if 'vp' not in st.session_state:
            st.session_state.vp = VideoProcessor()
        if 'running' not in st.session_state:
            st.session_state.running = False
        if 'summary_data' not in st.session_state:
            st.session_state.summary_data = None
        if 'json_ready' not in st.session_state:
            st.session_state.json_ready = False
        if 'manual_edit_mode' not in st.session_state:
            st.session_state.manual_edit_mode = False
            
        self.vp: VideoProcessor = st.session_state.vp
        
        # Restore hardcoded settings, removing the dynamic sidebar
        self.source = "output_filtered_3.mp4"
        self.location = "انبار سنگین"  # Heavy Warehouse in Persian
        self.weights = WarehouseConfig.WEIGHTS_DEFAULT
        self.model_input_size = WarehouseConfig.MODEL_INPUT_SIZE_DEFAULT
        self.line_x = WarehouseConfig.LINE_X_DEFAULT
        self.skip = WarehouseConfig.FRAME_SKIP_DEFAULT
        self.iou = WarehouseConfig.IOU_THRESH_DEFAULT
        self.conf = WarehouseConfig.CONF_THRESH_DEFAULT

    def render(self):
        """Render the complete Streamlit application."""
        st.set_page_config(
            page_title="Smart Warehouse CV Dashboard", 
            page_icon="📹", 
            layout="wide"
        )
        st.title("📹 Smart Warehouse CV Dashboard")
        
        self._render_sidebar()

        # Main application flow controller
        if st.session_state.running:
            self._render_live_display()
        elif st.session_state.summary_data:
            self._render_confirmation_interface(st.session_state.summary_data)
        else:
            # Show tabs only when not running and no summary is displayed
            prod_tab, demo_tab = st.tabs(["Production (Live)", "Demo (File)"])

            with prod_tab:
                self._render_processing_interface(is_demo=False)

            with demo_tab:
                self._render_processing_interface(is_demo=True)

    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            self.line_x = st.number_input(
                "Counting Line (X-coordinate)", 
                min_value=0, 
                max_value=1920, # Assuming max video width
                value=WarehouseConfig.LINE_X_DEFAULT, 
                step=10,
                help="Set the vertical line for counting. Objects crossing this line will be logged."
            )

    def _render_processing_interface(self, is_demo: bool):
        """
        Render the main processing interface for either live or demo mode.
        
        Args:
            is_demo: True if rendering for the demo tab, False for production.
        """
        if is_demo:
            source = "output_filtered_3.mp4"
            st.info(f"**Demo Mode:** Processing the local file `{source}`.")
        else:
            use_pi_camera = st.checkbox("Use Raspberry Pi Camera", value=True, help="If checked, the application will attempt to use the default camera connected to this Raspberry Pi.")
            
            if use_pi_camera:
                source = "picamera" # Special identifier for the video processor
            else:
                source = st.text_input(
                    "Video Source (URL or File Path)",
                    "http://192.168.144.170:5000/video_feed",
                    key="stream_url",
                    help="Enter a network stream URL (RTSP/HTTP) or a path to a local video file."
                )

        # The "Start" button is the only control needed here now
        if st.button(
            "▶️ Start Processing", 
            disabled=st.session_state.running, 
            key=f"start_{'demo' if is_demo else 'prod'}"
        ):
            st.session_state.running = True
            st.session_state.json_ready = False
            st.session_state.summary_data = None
            st.session_state.manual_edit_mode = False
            
            self.vp.start_processing(
                source, self.weights, self.line_x, 
                self.skip, self.iou, self.conf, self.location,
                self.model_input_size
            )
            st.rerun()

    def _render_live_display(self):
        """Render live video display with real-time statistics and event table."""
        # Stop button at the top of the live view
        if st.button("⏹️ Stop Processing", key="stop_live"):
            summary = self.vp.stop_processing()
            st.session_state.summary_data = summary
            st.session_state.running = False
            st.rerun()

        # Improved layout with columns
        main_view, side_info = st.columns([3, 2])

        with main_view:
            frame_placeholder = st.empty()
        
        with side_info:
            stats_placeholder = st.empty()
            st.subheader("Live Event Log")
            events_placeholder = st.empty()

        while st.session_state.running and self.vp.is_running:
            # Display latest frame
            if self.vp.latest_frame is not None:
                frame_placeholder.image(
                    cv2.cvtColor(self.vp.latest_frame, cv2.COLOR_BGR2RGB), 
                    use_container_width=True
                )
            
            # Display current statistics
            counts = self.vp.tracker.counts
            fps = self.vp.processing_fps
            with stats_placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Loaded/Unloaded", 
                        f"{counts['in']} / {counts['out']}", 
                        f"📍 {self.location}"
                    )
                with col2:
                    st.metric("Processing Speed", f"{fps:.1f} FPS")

            # Display live event table
            events = self.vp.tracker.events
            if events:
                df = pd.DataFrame(
                    events, 
                    columns=['Timestamp', 'Status', 'Track ID', 'Location', 'Product Type']
                )
                df = df.sort_values(by='Timestamp', ascending=False).reset_index(drop=True)
                events_placeholder.dataframe(df, use_container_width=True)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.02)
        
        # Handle case where processing stopped unexpectedly
        if st.session_state.running:
            st.session_state.running = False
            st.warning("Processing stopped.")
            st.rerun()

    def _handle_confirm_click(self, summary: Dict):
        """
        Handle confirmation button click - saves JSON and resets state.
        
        Args:
            summary: Session summary dictionary
        """
        # Prepare data payload
        payload = {
            "total_products": summary["total_products"],
            "operation_type": summary["operation_type"],
            "start_time": summary["start_time"],
            "end_time": summary["end_time"],
            "location": summary["location"],
            "events": summary["events"]
        }
        
        # Generate filename with timestamp
        filename = f"events_{datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y%m%d_%H%M%S')}.json"
        json_data = json.dumps(payload, indent=4)

        # Save to file
        try:
            with open(filename, "w") as f:
                f.write(json_data)
            st.success(f"✅ Summary saved successfully to {filename}!")
        except Exception as e:
            st.error(f"⚠️ Failed to save file: {e}")

        # Reset state
        st.session_state["summary_data"] = None
        st.session_state["json_ready"] = False
        st.session_state["manual_edit_mode"] = False

    def _handle_manual_edit_save(self, original_summary: Dict, edited_events: Dict):
        """
        Handle manual edit save - updates summary with corrected events and saves JSON.
        
        Args:
            original_summary: Original session summary dictionary
            edited_events: Dictionary of edited events with corrected product types
        """
        # Reconstruct events list from edited data
        updated_events = []
        for event_idx, event_data in edited_events.items():
            updated_events.append((
                event_data['timestamp'],
                event_data['status'], 
                event_data['track_id'],
                event_data['location'],
                event_data['product_type']
            ))
        
        # Recalculate detailed product counts from edited events
        detailed_product_counts = {"loaded": {}, "unloaded": {}}
        loaded_count = 0
        unloaded_count = 0
        
        for event_data in updated_events:
            status = event_data[1]      # 'loaded' or 'unloaded'
            product_type = event_data[4] # e.g., 'neshaste', 'sulfat'
            
            if status == 'loaded':
                loaded_count += 1
            elif status == 'unloaded':
                unloaded_count += 1
                
            if status in detailed_product_counts:
                detailed_product_counts[status][product_type] = detailed_product_counts[status].get(product_type, 0) + 1
        
        # Determine operation type
        total_products = max(loaded_count, unloaded_count)
        if loaded_count > unloaded_count:
            operation_type = "loaded"
        elif unloaded_count > loaded_count:
            operation_type = "unloaded" 
        elif loaded_count == unloaded_count and loaded_count > 0:
            operation_type = "balanced"
        else:
            operation_type = "none"
        
        # Format events for output (same structure as automatic detection)
        formatted_events = {
            str(i): {
                "timestamp": event[0],
                "status": event[1],
                "track_id": event[2],
                "location": event[3],
                "product_type": event[4]
            }
            for i, event in enumerate(updated_events)
        }
        
        # Create updated summary with same structure as automatic detection
        updated_summary = {
            "total_products": total_products,
            "operation_type": operation_type,
            "start_time": original_summary.get("start_time", datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')),
            "end_time": original_summary.get("end_time", datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')),
            "location": original_summary.get("location", self.location),
            "detailed_product_counts": detailed_product_counts,
            "events": formatted_events
        }
        
        # Prepare data payload (identical structure to automatic detection)
        payload = updated_summary.copy()
        payload["manual_edit"] = True  # Only difference - flag for manual edit
        
        # Generate filename with timestamp
        filename = f"events_manual_{datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y%m%d_%H%M%S')}.json"
        json_data = json.dumps(payload, indent=4, ensure_ascii=False)

        # Save to file
        try:
            with open(filename, "w", encoding='utf-8') as f:
                f.write(json_data)
            st.success(f"✅ Manual edit saved successfully to {filename}!")
        except Exception as e:
            st.error(f"⚠️ Failed to save file: {e}")

        # Reset state
        st.session_state["summary_data"] = None
        st.session_state["json_ready"] = False
        st.session_state["manual_edit_mode"] = False

    def _render_manual_edit_form(self, summary: Dict):
        """
        Render manual editing form for correcting individual detection events.
        
        Args:
            summary: Session summary dictionary
        """
        st.markdown("---")
        st.subheader("✏️ ویرایش دستی - اصلاح تشخیص محصولات")
        st.info("برای هر رویداد تشخیص (عبور فرک‌لیفت از خط)، نوع محصول صحیح را انتخاب کنید.")
        
        # Product types from config
        PRODUCT_TYPES = {
            "neshaste": "نشاسته",
            "pack_material": "مواد پک", 
            "sulfat": "سولفات آلومینیوم",
        }
        
        # Get events from summary
        events = summary.get("events", {})
        
        if not events:
            st.warning("هیچ رویدادی برای ویرایش وجود ندارد.")
            if st.button("🔙 بازگشت", key="back_to_summary_no_events"):
                st.session_state.manual_edit_mode = False
                st.rerun()
            return
        
        st.markdown(f"**تعداد کل رویدادها:** {len(events)}")
        st.markdown("---")
        
        # Create form for editing events
        edited_events = {}
        
        # Display events in chronological order
        sorted_events = sorted(events.items(), key=lambda x: x[1]['timestamp'])
        
        for event_idx, (event_id, event_data) in enumerate(sorted_events):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.write(f"**رویداد {event_idx + 1}**")
                    st.write(f"⏰ {event_data['timestamp']}")
                
                with col2:
                    # Status (loaded/unloaded) - Read only display
                    status_persian = "بارگیری" if event_data['status'] == 'loaded' else "تخلیه"
                    direction_icon = "📦 ➡️" if event_data['status'] == 'loaded' else "📤 ⬅️"
                    st.write(f"**نوع عملیات:**")
                    st.write(f"{direction_icon} {status_persian}")
                
                with col3:
                    st.write(f"**شناسه ردیابی:** {event_data['track_id']}")
                    st.write(f"**مکان:** {event_data['location']}")
                
                with col4:
                    # Product type selection
                    current_product = event_data.get('product_type', 'neshaste')
                    
                    # Find current product index for selectbox
                    product_options = list(PRODUCT_TYPES.keys())
                    try:
                        current_index = product_options.index(current_product)
                    except ValueError:
                        current_index = 0
                    
                    selected_product = st.selectbox(
                        "نوع محصول:",
                        options=product_options,
                        format_func=lambda x: PRODUCT_TYPES[x],
                        index=current_index,
                        key=f"product_{event_id}"
                    )
                    
                    # Store edited event
                    edited_events[event_id] = {
                        'timestamp': event_data['timestamp'],
                        'status': event_data['status'],
                        'track_id': event_data['track_id'],
                        'location': event_data['location'],
                        'product_type': selected_product
                    }
                
                st.markdown("---")
        
        # Show summary of changes
        st.markdown("#### 📊 خلاصه ویرایش‌ها")
        
        # Calculate new totals
        loaded_counts = {}
        unloaded_counts = {}
        total_loaded = 0
        total_unloaded = 0
        
        for event_data in edited_events.values():
            product_type = event_data['product_type']
            product_name = PRODUCT_TYPES[product_type]
            
            if event_data['status'] == 'loaded':
                total_loaded += 1
                loaded_counts[product_type] = loaded_counts.get(product_type, 0) + 1
            else:
                total_unloaded += 1
                unloaded_counts[product_type] = unloaded_counts.get(product_type, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 بارگیری:**")
            if loaded_counts:
                for product_key, count in loaded_counts.items():
                    st.write(f"- {count} عدد {PRODUCT_TYPES[product_key]}")
            else:
                st.write("- هیچ موردی")
            st.metric("مجموع بارگیری", total_loaded)
        
        with col2:
            st.markdown("**📤 تخلیه:**") 
            if unloaded_counts:
                for product_key, count in unloaded_counts.items():
                    st.write(f"- {count} عدد {PRODUCT_TYPES[product_key]}")
            else:
                st.write("- هیچ موردی")
            st.metric("مجموع تخلیه", total_unloaded)
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ ذخیره ویرایش دستی", key="save_manual_edit"):
                self._handle_manual_edit_save(summary, edited_events)
                st.rerun()
        
        with col2:
            if st.button("🔙 بازگشت", key="back_to_summary"):
                st.session_state.manual_edit_mode = False
                st.rerun()

    def _render_confirmation_interface(self, summary: Dict):
        """
        Render session summary and confirmation interface.
        
        Args:
            summary: Session summary dictionary
        """
        # Check if in manual edit mode
        if st.session_state.manual_edit_mode:
            self._render_manual_edit_form(summary)
            return
            
        st.markdown("---")
        st.subheader("📊 Session Summary")

        # Handle empty session
        if summary["total_products"] == 0:
            st.info("ℹ️ No products tracked. Resetting.")
            st.session_state.summary_data = None
            time.sleep(1)
            st.rerun()
            return

        # Define Persian translations
        PRODUCT_TRANSLATIONS_PERSIAN = {
            "neshaste": "نشاسته",
            "pack_material": "مواد پک",
            "sulfat": "سولفات آلومینیوم",
        }

        LOCATION_TRANSLATIONS_PERSIAN = {
            "Offline Warehouse": "انبار سنگین",
            "Production Warehouse A": "انبار تولید A",
            "Heavy Warehouse B": "انبار سنگین B",
            "Packing Warehouse C": "انبار بسته بندی C",
            "Storage Warehouse D": "انبار ذخیره سازی D",
        }
        
        OPERATION_TYPE_TRANSLATIONS_PERSIAN = {
            "loaded": "بارگیری",
            "unloaded": "تخلیه",
            "balanced": "متوازن",
            "none": "بدون عملیات",
        }

        # Construct detailed confirmation message in Persian as a list of lines
        confirmation_lines = ["#### 📋 **تاییدیه عملیات**"]

        raw_location = summary.get('location', 'نامشخص')
        confirmation_lines.append(f"**مکان:** {raw_location}")

        start_time_persian = summary.get('start_time', 'نامشخص')
        end_time_persian = summary.get('end_time', 'نامشخص')
        confirmation_lines.append(f"**بازه زمانی:** از {start_time_persian} تا {end_time_persian}")
        confirmation_lines.append("---")

        has_detailed_counts = False
        if summary.get("detailed_product_counts"):
            loaded_details = summary["detailed_product_counts"].get("loaded", {})
            unloaded_details = summary["detailed_product_counts"].get("unloaded", {})

            if loaded_details:
                has_detailed_counts = True
                confirmation_lines.append("**موارد بارگیری شده:**")
                for product, count in loaded_details.items():
                    product_persian = PRODUCT_TRANSLATIONS_PERSIAN.get(product, product)
                    confirmation_lines.append(f"- {count} عدد {product_persian}")
            
            if unloaded_details:
                has_detailed_counts = True
                if loaded_details: # Add a blank line if both loaded and unloaded exist and loaded had entries
                    if confirmation_lines[-1] != "**موارد بارگیری شده:**": # check if previous line was a product
                         confirmation_lines.append(" ") 
                confirmation_lines.append("**موارد تخلیه شده:**")
                for product, count in unloaded_details.items():
                    product_persian = PRODUCT_TRANSLATIONS_PERSIAN.get(product, product)
                    confirmation_lines.append(f"- {count} عدد {product_persian}")

        if not has_detailed_counts: # Fallback if no detailed counts found in detailed_product_counts
            total_products_overall = summary.get('total_products', 0)
            raw_operation_type = summary.get('operation_type', 'نامشخص')
            operation_type_persian = OPERATION_TYPE_TRANSLATIONS_PERSIAN.get(raw_operation_type, raw_operation_type)
            
            confirmation_lines.append("**خلاصه کلی عملیات:**")
            confirmation_lines.append(f"- تعداد کل محصولات: {total_products_overall}")
            confirmation_lines.append(f"- نوع عملیات: {operation_type_persian}")
        
        confirmation_lines.append("---")

        # Display the formatted confirmation message
        for line in confirmation_lines:
            st.markdown(line)
        
        st.warning("آیا اطلاعات فوق مورد تایید است؟") # General confirmation question
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.button(
                "✅ تایید و ذخیره",  # Confirm & Save in Persian
                key="confirm_save",
                on_click=self._handle_confirm_click,
                args=(summary,)
            )
        
        with col2:
            if st.button("✏️ ویرایش دستی", key="manual_edit"): # Manual Edit in Persian
                st.session_state.manual_edit_mode = True
                st.rerun()