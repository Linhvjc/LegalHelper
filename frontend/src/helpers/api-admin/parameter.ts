import axios from 'axios';
const instance = axios.create({
  baseURL: 'http://127.0.0.1:8000'
});

export const getParameters = async () => {
  const res = await instance.get("/parameter/get_all");
  console.log("Status: ", res.status)
  if (res.status !== 200) {
    throw new Error("Unable to fetch parameter");
  }
  const data = await res.data;
  return data;
};

export const deleteParameter = async (id: string) => {
  const res = await instance.delete(`/parameter/delete/${id}`);
  if (res.status !== 200) {
    throw new Error("Unable to delete parameter");
  }
  const data = await res.data;
  return data;
};

export const setSelectedParameter = async (id: string) => {
  const res = await instance.post(`/parameter/set_selected/${id}`);
  if (res.status !== 200) {
    throw new Error("Unable to set selected parameter");
  }
  const data = await res.data;
  return data;
};

export const addParameter = async (
  name: string,
  retrieverModel: string,
  generativeModel: string,
  databasePath: string,
  retrievalMaxLength: number,
  isSelected: boolean
) => {
  try {
    const res = await instance.post("/parameter/add", {
      name,
      retrieverModel,
      generativeModel,
      databasePath,
      retrievalMaxLength,
      isSelected
    });

    if (res.status !== 200) {
      throw new Error("Unable to add parameter");
    }

    const data = await res.data;
    return data;
  } catch (error) {
    throw new Error(`Error: ${error}`);
  }
};