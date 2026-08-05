# What the eye cameras see

Raw frames from the two inward-facing cameras, with the tracker's lock drawn on.

These are the input to the whole glasses-on-face measurement. The green box is where the
model thinks the inner eye corner is. The dark shape at the top of each frame is the nose
bridge support, which blocks the top third of the view and is the reason the search band
sits below it.

The left and right frames look different because the room is lit from one side. That
asymmetry is not a fault, it is the normal case, and it is why every frame is brightness
normalised before the model sees it.
