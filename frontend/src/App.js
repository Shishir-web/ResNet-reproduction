import { useRef, useState } from "react";
import axios from "axios";

export default function App() {
  const [image, setImage]     = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      setPreview(ev.target.result);
      setImage(ev.target.result.split(',')[1]);
      setResults(null);
    };
    reader.readAsDataURL(file);
  };

  const predict = async () => {
    if (!image) return;
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5001/predict',
        { image });
      setResults(res.data.predictions);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  return (
    <div style={{ minHeight:'100vh', background:'#0f0f0f',
      display:'flex', flexDirection:'column', alignItems:'center',
      justifyContent:'center', fontFamily:'monospace', padding:'24px' }}>

      <h1 style={{ color:'#fff', marginBottom:'6px' }}>
        ResNet-18 Image Classifier
      </h1>
      <p style={{ color:'#888', fontSize:'13px', marginBottom:'28px' }}>
        Reproduction of He et al. (2015) — Tiny-ImageNet 200 classes
      </p>

      <div onClick={() => inputRef.current.click()}
        style={{ width:'320px', height:'220px', border:'2px dashed #333',
          borderRadius:'10px', display:'flex', alignItems:'center',
          justifyContent:'center', cursor:'pointer', overflow:'hidden',
          background: preview ? 'transparent' : '#1a1a1a' }}>
        {preview
          ? <img src={preview} alt="upload"
              style={{ width:'100%', height:'100%', objectFit:'cover' }}/>
          : <span style={{ color:'#555', fontSize:'13px' }}>
              Click to upload an image
            </span>
        }
      </div>
      <input ref={inputRef} type="file" accept="image/*"
        onChange={handleFile} style={{ display:'none' }}/>

      <button onClick={predict} disabled={!image || loading}
        style={{ marginTop:'16px', padding:'10px 32px',
          background: image ? '#fff' : '#333', color: image ? '#000' : '#666',
          border:'none', borderRadius:'6px', cursor: image ? 'pointer' : 'default',
          fontWeight:'bold', fontFamily:'monospace' }}>
        {loading ? 'Classifying...' : 'Classify'}
      </button>

      {results && (
        <div style={{ marginTop:'28px', width:'320px' }}>
          {results.map((r, i) => (
            <div key={i} style={{ marginBottom:'12px' }}>
              <div style={{ display:'flex', justifyContent:'space-between',
                color: i === 0 ? '#fff' : '#888', fontSize:'13px',
                marginBottom:'4px' }}>
                <span>{r.class.replace(/_/g, ' ')}</span>
                <span>{(r.confidence * 100).toFixed(1)}%</span>
              </div>
              <div style={{ background:'#222', borderRadius:'4px',
                height:'4px', overflow:'hidden' }}>
                <div style={{ width:`${r.confidence * 100}%`,
                  height:'100%',
                  background: i === 0 ? '#fff' : '#444',
                  transition:'width 0.4s ease' }}/>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}