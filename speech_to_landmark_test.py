from speech_to_landmark import SpeechToLandmarkConfig, SpeechToLandmark


config = SpeechToLandmarkConfig(batch_size=2,
                                max_length=15000,
                                dataset_dir=r"D:\ObamaWeeklyAddress\tfrecord",
                                save_network_frequency=10,
                                speech_content_model_dir="speech_content_checkpoints",
                                use_speaker_aware=False,
                                is_2d=True, # Không có tác dụng gì trong main function
                                is_loadmodel=False,
                                learning_rate=1e-4)

speech_to_landmark = SpeechToLandmark(config=config)

speech_to_landmark.train()
