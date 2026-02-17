import { useCallback, useState } from 'react';

export interface Transform {
  scale: number;
  x: number;
  y: number;
}

export const useCanvasTransform = (initialScale = 1) => {
  const [transform, setTransform] = useState<Transform>({ scale: initialScale, x: 0, y: 0 });

  const handleZoom = useCallback((deltaY: number, mouseX: number, mouseY: number) => {
    setTransform(prev => {
      const zoomSpeed = 0.001;
      const newScale = Math.min(Math.max(prev.scale - deltaY * zoomSpeed, 0.1), 20);
      const scaleChange = newScale / prev.scale;

      return {
        scale: newScale,
        x: mouseX - (mouseX - prev.x) * scaleChange,
        y: mouseY - (mouseY - prev.y) * scaleChange
      };
    });
  }, []);

  const handlePan = useCallback((dx: number, dy: number) => {
    setTransform(prev => ({
      ...prev,
      x: prev.x + dx,
      y: prev.y + dy
    }));
  }, []);

  return { transform, setTransform, handleZoom, handlePan };
};
