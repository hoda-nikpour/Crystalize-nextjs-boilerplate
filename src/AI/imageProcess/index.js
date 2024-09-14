// src/components/ClickedImages.js
import React from 'react';
import { useImageClickContext } from 'lib/ImageClickContext';
import AnalyzedByGenerativeAI from './AnalyzedByGenerativeAI';
import AnalyzedByTraditionalAI from './AnalyzedByTraditionalAI';

const ClickedImages = () => {
  const { clickedImageIds } = useImageClickContext();
  const img1 = '/static/img1.jpg';
  const img2 = '/static/img2.jpg';
  const img3 = '/static/img3.jpg';

  return (
    <div>
      {clickedImageIds.map((id, index) => {
        if (
          (clickedImageIds[index] === '610808511d4f130008d41145' &&
            clickedImageIds[index + 1] === '6108088f32791a000b483f29') ||
          (clickedImageIds[index] === '6108088f32791a000b483f29' &&
            clickedImageIds[index + 1] === '610808511d4f130008d41145')
        ) {
          return (
            <AnalyzedByGenerativeAI
              customkey={1}
              image1={img1}
              image2={img2}
              setNum={1}
            />
          );
        }
        return null;
      })}

      {clickedImageIds.map((id, index) => {
        if (
          (clickedImageIds[index] === '6108088f32791a000b483f29' &&
            clickedImageIds[index + 1] === '6108085b1d4f130008d41148') ||
          (clickedImageIds[index] === '6108085b1d4f130008d41148' &&
            clickedImageIds[index + 1] === '6108088f32791a000b483f29')
        ) {
          return (
            <AnalyzedByGenerativeAI
              customkey={2}
              image1={img2}
              image2={img3}
              setNum={2}
            />
          );
        }
        return null;
      })}

      {clickedImageIds.map((id, index) => {
        if (
          (clickedImageIds[index] === '6108085b1d4f130008d41148' &&
            clickedImageIds[index + 1] === '610808511d4f130008d41145') ||
          (clickedImageIds[index] === '610808511d4f130008d41145' &&
            clickedImageIds[index + 1] === '6108085b1d4f130008d41148')
        ) {
          return (
            <AnalyzedByTraditionalAI
              customkey={3}
              image1={img1}
              image2={img3}
            />
          );
        }
        return null;
      })}
    </div>
  );
};

export default ClickedImages;
