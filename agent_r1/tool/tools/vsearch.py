from PIL import Image
from typing import Dict, List, Any
from agent_r1.tool.tool_base import Tool

class VisualSearchTool(Tool):
    """
        Tool for visual QA, crop the image
    """
    def __init__(self):
        name = "image_crop"
        description = "Crop and enlarge the picture, which is used when you need to focus on a certain area of the picture. Input the boundbox of the area"
        parameters = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "x-coordinate of the top-left corner of the bounding box"
                },

                "y": {
                    "type": "number",
                    "description": "y-coordinate of the top-left corner of the bounding box"
                },

                "w": {
                    "type": "number",
                    "description": "width of the bounding box"
                },

                "h": {
                    "type": "number",
                    "description": "height of the bounding box "
                },
                
            },
            "required": ["x", "y", "w", "h"]
        }
        
        super().__init__(name, description, parameters)
    

    def execute(self, args: Dict, multi_modal_data = None) -> Dict[str, Any]:
        """
        Execute the tool functionality
        
        Args:
            args: Tool parameters

        Returns:
            Dictionary containing:
            - content: str - Text content of the result
            - image: Optional[bytes] - Image data if applicable
        """
        result = {}
        if multi_modal_data is not None:
            image: Image = multi_modal_data['image'][0]
        
            # crop the image
            x, y, w, h = args['x'], args['y'], args['w'], args['h']
            
            try:
                cropped_image = image.crop((x, y, x + w, y + h))
                result['image'] = cropped_image
                result['content'] = "The cropped image is generated, analysis the cropped image <image>"

            except Exception as _:
                result['image'] = None
                result['content'] = "CropError: crop failed, please check if the bbox value are valid"
            

        else:
            raise RuntimeError("Error: Image is none")

        return result

    def calculate_reward(self, args: Dict, result: Dict[str, Any], ground_truth_tool = None) -> float:
        """iou based reward"""
        reward = 0.0
        x, y, w, h = args['x'], args['y'], args['w'], args['h']
        
        # reward bbox in a valid range. 
        if result.startswith("CropError"):
            reward += 0.01  # Small reward for trying
        
        else:
            # reward bbox in iou
            iou = self._iou(ground_truth_tool, [x,y,w,h])
            
            # clip the iou
            if iou <= 0.3:
                iou = 0.0
            reward += iou

        return reward

    def _iou(self, gt_bbox, gen_bbox):
        """calculate the iou of two bounding boxes 0~1"""
        gt_x, gt_y, gt_w, gt_h = gt_bbox
        x, y, w, h = gen_bbox
        inter_x_min = max(gt_x, x)
        inter_y_min = max(gt_y, y)
        inter_x_max = min(gt_x + gt_w, x + w)
        inter_y_max = min(gt_y + gt_h, y + h)

        inter_width = inter_x_max - inter_x_min
        inter_height = inter_y_max - inter_y_min
        if inter_width <= 0 or inter_height <= 0:
            return 0.0
        inter_area = inter_width * inter_height

        gt_area = gt_w * gt_h
        pred_area = w * h

        union_area = gt_area + pred_area - inter_area
        if union_area == 0:
            return 0.0