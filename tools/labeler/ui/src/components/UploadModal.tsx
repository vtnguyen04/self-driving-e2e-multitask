import { FileArchive, FolderArchive, Loader2, Upload, Video, X } from 'lucide-react';
import React, { useState } from 'react';

interface UploadModalProps {
    projectId: number;
    onClose: () => void;
    onSuccess: () => void;
}

type UploadTab = 'images' | 'video' | 'folder' | 'export';

export const UploadModal: React.FC<UploadModalProps> = ({ projectId, onClose, onSuccess }) => {
    const [activeTab, setActiveTab] = useState<UploadTab>('images');
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState('');
    const [sampleRate, setSampleRate] = useState(5);

    const handleImagesUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        setUploading(true);
        setProgress(`Uploading ${files.length} images...`);

        try {
            const formData = new FormData();
            formData.append('project_id', projectId.toString());
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }

            const response = await fetch('/api/v1/upload/images', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            setProgress(`✅ Uploaded ${result.uploaded_count} images`);
            setTimeout(() => {
                onSuccess();
                onClose();
            }, 1500);
        } catch (error) {
            setProgress(`❌ Upload failed: ${error}`);
        } finally {
            setUploading(false);
        }
    };

    const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);
        setProgress(`Extracting frames from video (${sampleRate} fps)...`);

        try {
            const formData = new FormData();
            formData.append('project_id', projectId.toString());
            formData.append('sample_rate', sampleRate.toString());
            formData.append('file', file);

            const response = await fetch('/api/v1/upload/video', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            setProgress(`✅ Extracted ${result.extracted_count} frames from ${result.total_frames} total`);
            setTimeout(() => {
                onSuccess();
                onClose();
            }, 1500);
        } catch (error) {
            setProgress(`❌ Upload failed: ${error}`);
        } finally {
            setUploading(false);
        }
    };

    const handleFolderUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);
        setProgress(`Extracting images from ZIP...`);

        try {
            const formData = new FormData();
            formData.append('project_id', projectId.toString());
            formData.append('file', file);

            const response = await fetch('/api/v1/upload/folder', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            setProgress(`✅ Uploaded ${result.uploaded_count} images from ZIP`);
            setTimeout(() => {
                onSuccess();
                onClose();
            }, 1500);
        } catch (error) {
            setProgress(`❌ Upload failed: ${error}`);
        } finally {
            setUploading(false);
        }
    };

    const handleExportUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);
        setProgress(`Importing labeled export...`);

        try {
            const formData = new FormData();
            formData.append('project_id', projectId.toString());
            formData.append('file', file);

            const response = await fetch('/api/v1/upload/export', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            setProgress(`✅ Imported ${result.imported_count} samples with labels`);
            setTimeout(() => {
                onSuccess();
                onClose();
            }, 1500);
        } catch (error) {
            setProgress(`❌ Import failed: ${error}`);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black/95 backdrop-blur-xl z-[100] flex items-center justify-center p-6">
            <div className="w-full max-w-3xl bg-[#0a0a0c] border-2 border-white/20 rounded-[3rem] overflow-hidden shadow-[0_0_150px_rgba(0,255,65,0.3)]">
                <div className="p-10 border-b border-white/10 flex justify-between items-center bg-white/5">
                    <h2 className="text-3xl font-black text-white flex items-center gap-5 font-cyber tracking-tighter">
                        <Upload className="w-10 h-10 text-accent" /> UPLOAD_DATA
                    </h2>
                    <button onClick={onClose} className="p-4 hover:bg-white/10 rounded-full transition-colors text-white">
                        <X className="w-10 h-10" />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-white/10 bg-white/[0.02]">
                    <TabButton icon={<Upload />} label="Images" active={activeTab === 'images'} onClick={() => setActiveTab('images')} />
                    <TabButton icon={<Video />} label="Video" active={activeTab === 'video'} onClick={() => setActiveTab('video')} />
                    <TabButton icon={<FolderArchive />} label="Folder ZIP" active={activeTab === 'folder'} onClick={() => setActiveTab('folder')} />
                    <TabButton icon={<FileArchive />} label="Import Export" active={activeTab === 'export'} onClick={() => setActiveTab('export')} />
                </div>

                {/* Content */}
                <div className="p-12 min-h-[400px]">
                    {activeTab === 'images' && (
                        <UploadSection
                            title="Upload Images"
                            description="Select multiple images (JPG, PNG, BMP)"
                            accept="image/*"
                            multiple
                            onChange={handleImagesUpload}
                            uploading={uploading}
                        />
                    )}

                    {activeTab === 'video' && (
                        <div className="space-y-8">
                            <div className="space-y-4">
                                <label className="text-white font-black text-lg">Sample Rate: {sampleRate} fps</label>
                                <input
                                    type="range"
                                    min="1"
                                    max="30"
                                    value={sampleRate}
                                    onChange={(e) => setSampleRate(parseInt(e.target.value))}
                                    className="w-full h-3 bg-white/10 rounded-full appearance-none cursor-pointer accent-accent"
                                    disabled={uploading}
                                />
                                <p className="text-white/60 text-sm">Extract {sampleRate} frames per second from video</p>
                            </div>
                            <UploadSection
                                title="Upload Video"
                                description="Select video file (MP4, AVI, MOV)"
                                accept="video/*"
                                onChange={handleVideoUpload}
                                uploading={uploading}
                            />
                        </div>
                    )}

                    {activeTab === 'folder' && (
                        <UploadSection
                            title="Upload Folder ZIP"
                            description="Select ZIP file containing images (auto-filtered)"
                            accept=".zip"
                            onChange={handleFolderUpload}
                            uploading={uploading}
                        />
                    )}

                    {activeTab === 'export' && (
                        <UploadSection
                            title="Import Labeled Export"
                            description="Select previously exported ZIP with labels (train/val/test structure)"
                            accept=".zip"
                            onChange={handleExportUpload}
                            uploading={uploading}
                        />
                    )}

                    {/* Progress */}
                    {progress && (
                        <div className="mt-8 p-6 bg-white/5 rounded-2xl border border-white/10">
                            <div className="flex items-center gap-4">
                                {uploading && <Loader2 className="w-6 h-6 text-accent animate-spin" />}
                                <p className="text-white font-bold text-base">{progress}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

const TabButton = ({ icon, label, active, onClick }: { icon: React.ReactNode, label: string, active: boolean, onClick: () => void }) => (
    <button
        onClick={onClick}
        className={`flex-1 flex items-center justify-center gap-3 p-6 text-base font-black transition-all ${
            active ? 'bg-accent/20 text-white border-b-4 border-accent' : 'text-white/40 hover:text-white hover:bg-white/5'
        }`}
    >
        <span className="w-6 h-6">{icon}</span>
        {label}
    </button>
);

const UploadSection = ({ title, description, accept, multiple, onChange, uploading }: {
    title: string,
    description: string,
    accept: string,
    multiple?: boolean,
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void,
    uploading: boolean
}) => (
    <div className="space-y-6">
        <div>
            <h3 className="text-2xl font-black text-white mb-2">{title}</h3>
            <p className="text-white/60 text-base">{description}</p>
        </div>
        <label className="block">
            <div className="border-4 border-dashed border-white/20 rounded-3xl p-16 text-center hover:border-accent/40 hover:bg-white/5 transition-all cursor-pointer">
                <Upload className="w-16 h-16 text-white/40 mx-auto mb-6" />
                <p className="text-white font-black text-lg mb-2">Click to browse files</p>
                <p className="text-white/40 text-sm">or drag and drop here</p>
            </div>
            <input
                type="file"
                accept={accept}
                multiple={multiple}
                onChange={onChange}
                disabled={uploading}
                className="hidden"
            />
        </label>
    </div>
);
