import cv2
import os
import pathlib
import torch

    
def process_recording(recording_path, save_folder):

    def split_video_into_frames(video_path, output_folder):

        # Initialize the video capture
        if isinstance(video_path, pathlib.Path):
            video_path = str(pathlib.PurePath(video_path))
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            # Read frame by frame
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no frames left

            # Save each frame as an image
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        # Release the video capture object
        cap.release()
        print(f"Total frames extracted: {frame_count}")
    
    movie_name = recording_path.stem
    if not os.path.exists(save_folder / movie_name):
        frames_folder = save_folder / movie_name / 'frames'
        frames_folder.mkdir(parents=True, exist_ok=False)
        audio_folder = save_folder / movie_name / 'audio'
        audio_folder.mkdir(parents=True, exist_ok=False)
    
    split_video_into_frames(recording_path, save_folder / movie_name / "frames/")
    # extract_audio_in_stereo(recording_path, save_folder / movie_name / 'audio/audio.mp3')
    # rename_frames_folder(save_folder / movie_name / 'frames/')
    
class RNNModule(torch.nn.Module):
    def __init__(self, device, n_inputs, n_hidden, nonlinearity, input_bias=True, hidden_bias=True):
        super(RNNModule, self).__init__()

        self.rnn_cell = RNNCell(n_inputs, n_hidden, nonlinearity, input_bias, hidden_bias)
        self.n_hidden = n_hidden

        self.device = device

    def forward(self, x, hidden=None):
        # x: [BATCH SIZE, TIME, N_FEATURES]
        
        output = torch.zeros(x.shape[0], x.shape[1], self.n_hidden).to(self.device)

        if hidden is None:
            h_out = torch.zeros(x.shape[0], self.n_hidden) # initialize hidden state
            h_out = h_out.to(self.device)
        else:
            h_out = hidden

        window_size = x.shape[1]

        # loop over time
        for t in range(window_size):
            x_t = x[:,t,...]
            h_out = self.rnn_cell(x_t, h_out)
            output[:,t,...] = h_out

        # return all outputs, and the last hidden state
        return output, h_out

class RNNCell(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, nonlinearity, input_bias, hidden_bias):
        super(RNNCell, self).__init__()

        if nonlinearity == 'sigmoid':
            activation_fn = torch.sigmoid
        else:
            print("[!!!] WARNING: activation function not recognized, using identity")
            activation_fn = torch.nn.Identity()

        self.in2hidden = torch.nn.Linear(n_inputs, n_hidden, bias=input_bias)
        self.hidden2hidden = torch.nn.Linear(n_hidden, n_hidden, bias=hidden_bias)

        self.activation_fn = activation_fn

    def forward(self, x, hidden):
        igates = self.in2hidden(x)
        hgates = self.hidden2hidden(hidden)
        return self.activation_fn(igates + hgates)
